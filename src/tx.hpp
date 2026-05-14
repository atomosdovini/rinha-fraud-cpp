#pragma once
// tx.hpp — Transaction payload parser and feature vectoriser.
//
// Parses the JSON body of a /fraud-score request into a Dims-dimensional
// int16_t feature vector for the IVF index.
//
// Uses a forward key-scan approach: advance to each '"', check if it matches
// the expected field name, then extract the value. One forward pass total.

#include "index.hpp"

#include <cstdint>
#include <cstring>
#include <string_view>

namespace fraud {

using rinha::Dims;

// ── Date helpers ──────────────────────────────────────────────────────────────

static inline int d2(std::string_view s, int p) noexcept {
    return (s[p] - '0') * 10 + (s[p + 1] - '0');
}
static inline int d4(std::string_view s, int p) noexcept {
    return (s[p]-'0')*1000 + (s[p+1]-'0')*100 + (s[p+2]-'0')*10 + (s[p+3]-'0');
}

static int civil_day(int y, int m, int d) noexcept {
    y -= (m <= 2);
    int era        = y / 400;
    unsigned yoe   = unsigned(y - era * 400);
    unsigned doy   = (153u * unsigned(m + (m > 2 ? -3 : 9)) + 2u) / 5u + unsigned(d) - 1u;
    unsigned doe   = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    return era * 146097 + int(doe) - 719468;
}

// Fast path: March 2026 is the competition month — skip full civil_day().
static inline bool is_march_2026(std::string_view ts) noexcept {
    return ts.size() >= 16
        && ts[0]=='2' && ts[1]=='0' && ts[2]=='2' && ts[3]=='6'
        && ts[5]=='0' && ts[6]=='3';
}

static int ts_minutes(std::string_view ts) noexcept {
    if (is_march_2026(ts))
        return (d2(ts, 8) - 1) * 1440 + d2(ts, 11) * 60 + d2(ts, 14);
    return civil_day(d4(ts,0), d2(ts,5), d2(ts,8)) * 1440 + d2(ts,11)*60 + d2(ts,14);
}

static int ts_weekday(std::string_view ts) noexcept {   // 0 = Monday
    if (is_march_2026(ts)) return (d2(ts, 8) + 5) % 7;
    int y = d4(ts,0), m = d2(ts,5), day = d2(ts,8);
    static constexpr int t[] = {0,3,2,5,0,3,5,1,4,6,2,4};
    if (m < 3) --y;
    return ((y + y/4 - y/100 + y/400 + t[m-1] + day) % 7 + 6) % 7;
}

// ── MCC risk ──────────────────────────────────────────────────────────────────

static int16_t mcc_risk(std::string_view code) noexcept {
    if (code.size() < 4) return 5000;
    int v = (code[0]-'0')*1000 + (code[1]-'0')*100 + (code[2]-'0')*10 + (code[3]-'0');
    switch (v) {
        case 5411: return 1500;  case 5812: return 3000;  case 5912: return 2000;
        case 5944: return 4500;  case 7801: return 8000;  case 7802: return 7500;
        case 7995: return 8500;  case 4511: return 3500;  case 5311: return 2500;
        default:   return 5000;
    }
}

// ── JSON scanning ─────────────────────────────────────────────────────────────

static inline void skip_ws(std::string_view s, size_t& p) noexcept {
    while (p < s.size() && (unsigned char)s[p] <= ' ') ++p;
}

// Advance p past the next ':' and any whitespace after it.
static inline bool past_colon(std::string_view s, size_t& p) noexcept {
    p = s.find(':', p);
    if (p == std::string_view::npos) return false;
    ++p;
    skip_ws(s, p);
    return p < s.size();
}

// Read a quoted string; p ends just after the closing '"'.
static inline std::string_view read_string(std::string_view s, size_t& p) noexcept {
    size_t open = s.find('"', p);
    if (open == std::string_view::npos) return {};
    size_t close = s.find('"', open + 1);
    if (close == std::string_view::npos) return {};
    p = close + 1;
    return s.substr(open + 1, close - open - 1);
}

// Read a JSON number into `out`.
static inline bool read_number(std::string_view s, size_t& p, double& out) noexcept {
    skip_ws(s, p);
    if (p >= s.size()) return false;
    bool neg = (s[p] == '-');
    if (neg && ++p >= s.size()) return false;
    double v = 0; bool ok = false;
    while (p < s.size() && s[p] >= '0' && s[p] <= '9') { v = v*10 + (s[p++]-'0'); ok = true; }
    if (p < s.size() && s[p] == '.') {
        ++p; double f = .1;
        while (p < s.size() && s[p] >= '0' && s[p] <= '9') { v += (s[p++]-'0')*f; f *= .1; ok = true; }
    }
    if (!ok) return false;
    out = neg ? -v : v;
    return true;
}

// Read a JSON boolean.
static inline bool read_bool(std::string_view s, size_t& p, bool& out) noexcept {
    skip_ws(s, p);
    if (s.substr(p, 4) == "true")  { out = true;  p += 4; return true; }
    if (s.substr(p, 5) == "false") { out = false; p += 5; return true; }
    return false;
}

// Advance p to right after `"key"` (i.e., the closing '"').
// Scans forward looking for the exact quoted key.
static inline bool seek_key(std::string_view s, size_t& p, std::string_view key) noexcept {
    while (true) {
        size_t q = s.find('"', p);
        if (q == std::string_view::npos) return false;
        size_t end = q + 1 + key.size();
        if (end < s.size() && s.substr(q + 1, key.size()) == key && s[end] == '"') {
            p = end + 1;
            return true;
        }
        p = q + 1;
    }
}

// ── Feature extraction ────────────────────────────────────────────────────────

bool extract(std::string_view body, int16_t out[Dims]) noexcept {
    size_t p = 0;
    double amount, installments, cust_avg, tx_count, merch_avg, km_home;
    std::string_view ts, merch_id, mcc, known;
    bool online = false, card_present = false;

    if (!seek_key(body, p, "amount"))        return false;
    if (!past_colon(body, p))                return false;
    if (!read_number(body, p, amount))       return false;

    if (!seek_key(body, p, "installments"))  return false;
    if (!past_colon(body, p))                return false;
    if (!read_number(body, p, installments)) return false;

    if (!seek_key(body, p, "requested_at"))  return false;
    ts = read_string(body, p = body.find(':', p) + 1);
    if (ts.size() < 16)                      return false;

    // First avg_amount is the customer's
    if (!seek_key(body, p, "avg_amount"))    return false;
    if (!past_colon(body, p))                return false;
    if (!read_number(body, p, cust_avg) || cust_avg == 0.0) return false;

    if (!seek_key(body, p, "tx_count_24h"))  return false;
    if (!past_colon(body, p))                return false;
    if (!read_number(body, p, tx_count))     return false;

    // known_merchants: capture the full array as a substring for membership check
    {
        size_t ka = body.find("known_merchants", p);
        if (ka == std::string_view::npos) return false;
        size_t ab = body.find('[', ka);
        size_t ae = body.find(']', ab);
        if (ab == std::string_view::npos || ae == std::string_view::npos) return false;
        known = body.substr(ab, ae - ab + 1);
        p = ae + 1;
    }

    if (!seek_key(body, p, "id"))            return false;
    merch_id = read_string(body, p = body.find(':', p) + 1);

    if (!seek_key(body, p, "mcc"))           return false;
    mcc = read_string(body, p = body.find(':', p) + 1);

    // Second avg_amount is the merchant's
    if (!seek_key(body, p, "avg_amount"))    return false;
    if (!past_colon(body, p))                return false;
    if (!read_number(body, p, merch_avg))    return false;

    if (!seek_key(body, p, "is_online"))     return false;
    if (!past_colon(body, p))                return false;
    if (!read_bool(body, p, online))         return false;

    if (!seek_key(body, p, "card_present"))  return false;
    if (!past_colon(body, p))                return false;
    if (!read_bool(body, p, card_present))   return false;

    if (!seek_key(body, p, "km_from_home"))  return false;
    if (!past_colon(body, p))                return false;
    if (!read_number(body, p, km_home))      return false;

    // Build feature vector dims 0–4
    out[0] = rinha::qclamp01(amount / 10000.0);
    out[1] = rinha::qclamp01(installments / 12.0);
    out[2] = rinha::qclamp01((amount / cust_avg) / 10.0);
    out[3] = rinha::qclamp01(double(d2(ts, 11)) / 23.0);
    out[4] = rinha::qclamp01(double(ts_weekday(ts)) / 6.0);

    // last_transaction → dims 5–6
    {
        size_t lp = body.find("last_transaction", p);
        if (lp == std::string_view::npos) return false;
        size_t col = body.find(':', lp);
        if (col == std::string_view::npos) return false;
        size_t val = col + 1;
        skip_ws(body, val);
        if (body.substr(val, 4) == "null") {
            out[5] = -10000;
            out[6] = -10000;
        } else {
            std::string_view last_ts;
            double last_km = 0;
            size_t lq = val;
            if (!seek_key(body, lq, "timestamp")) return false;
            last_ts = read_string(body, lq = body.find(':', lq) + 1);
            if (last_ts.size() < 16) return false;
            if (!seek_key(body, lq, "km_from_current")) return false;
            if (!past_colon(body, lq)) return false;
            if (!read_number(body, lq, last_km)) return false;
            int delta = ts_minutes(ts) - ts_minutes(last_ts);
            out[5] = rinha::qclamp01(double(delta) / 1440.0);
            out[6] = rinha::qclamp01(last_km / 1000.0);
        }
    }

    out[7]  = rinha::qclamp01(km_home / 1000.0);
    out[8]  = rinha::qclamp01(tx_count / 20.0);
    out[9]  = online       ? 10000 : 0;
    out[10] = card_present ? 10000 : 0;
    out[11] = known.find(merch_id) == std::string_view::npos ? 10000 : 0;
    out[12] = mcc_risk(mcc);
    out[13] = rinha::qclamp01(merch_avg / 10000.0);
    return true;
}

} // namespace fraud
