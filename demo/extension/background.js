// Service worker: nhận yêu cầu từ content script, gọi máy chủ dịch (tránh chặn CORS/mixed content của trang https).
// Địa chỉ máy chủ và khóa lưu trong chrome.storage.local (đặt ở popup).
const MAC_DINH = { may_chu: "http://127.0.0.1:8765", khoa: "" };
const HAN_MS = { "/dich": 180000, "/suc_khoe": 15000 };

// Đường hầm quick tunnel thỉnh thoảng đứt kết nối ngẫu nhiên: lỗi mạng (không phải quá hạn, không phải HTTP) thì thử lại tối đa 2 lần.
// An toàn vì yêu cầu dịch là xác định (greedy) và máy chủ có bộ đệm.
async function goi(duong_dan, body) {
  let r;
  for (let lan = 0; lan < 3; lan++) {
    r = await goi_mot_lan(duong_dan, body);
    if (r.ok || !r.loi_mang) return r;
    await new Promise((x) => setTimeout(x, 1500 * (lan + 1)));
  }
  return r;
}

async function goi_mot_lan(duong_dan, body) {
  const c = await chrome.storage.local.get(MAC_DINH);
  const url = c.may_chu.replace(/\/+$/, "") + duong_dan;
  const ac = new AbortController();
  const tm = setTimeout(() => ac.abort(), HAN_MS[duong_dan] || 60000);
  try {
    const headers = { "Content-Type": "application/json" };
    if (c.khoa) headers["X-Demo-Key"] = c.khoa;
    const r = await fetch(url, { method: body ? "POST" : "GET", headers, body: body ? JSON.stringify(body) : undefined, signal: ac.signal });
    if (r.status === 401) return { ok: false, loi: "Sai khóa truy cập (đặt lại trong popup của extension)." };
    const data = await r.json().catch(() => null);
    if (!r.ok) return { ok: false, loi: `Máy chủ báo lỗi ${r.status}${data && data.loi ? ": " + data.loi : ""}` };
    return { ok: true, data };
  } catch (e) {
    const het_han = e.name === "AbortError";
    return { ok: false, loi_mang: !het_han, loi: het_han ? "Máy chủ trả lời quá lâu (quá hạn)." : "Không kết nối được máy chủ dịch (kiểm tra địa chỉ trong popup và máy chủ đã chạy chưa)." };
  } finally {
    clearTimeout(tm);
  }
}

// 1688 chọn ngôn ngữ nội dung theo cookie oversealanguage: đặt 'zh' để trang trả bản tiếng Trung của người bán (đúng thứ mô hình dịch).
async function ep_tieng_trung() {
  try {
    await chrome.cookies.set({ url: "https://detail.1688.com/", domain: ".1688.com", name: "oversealanguage", value: "zh", path: "/", secure: true });
    return { ok: true };
  } catch (e) {
    return { ok: false, loi: "Không đặt được cookie ngôn ngữ: " + e };
  }
}

chrome.runtime.onMessage.addListener((m, _sender, tra_loi) => {
  if (m && m.loai_yeu_cau === "ep_zh") ep_tieng_trung().then(tra_loi);
  else if (m && m.loai_yeu_cau === "dich") goi("/dich", m.body).then(tra_loi);
  else if (m && m.loai_yeu_cau === "suc_khoe") goi("/suc_khoe", null).then(tra_loi);
  else return false;
  return true;      // trả lời bất đồng bộ
});
