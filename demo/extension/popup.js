const MAC_DINH = { may_chu: "http://127.0.0.1:8765", khoa: "" };
const $ = (id) => document.getElementById(id);

function hien(mau, chu) {
  $("den").style.background = mau;
  $("chu").textContent = chu;
}

async function nap() {
  const c = await chrome.storage.local.get(MAC_DINH);
  $("may_chu").value = c.may_chu;
  $("khoa").value = c.khoa;
}

async function luu() {
  await chrome.storage.local.set({ may_chu: $("may_chu").value.trim() || MAC_DINH.may_chu, khoa: $("khoa").value.trim() });
}

async function kiem() {
  await luu();
  hien("#c9a227", "Đang kiểm tra…");
  const r = await chrome.runtime.sendMessage({ loai_yeu_cau: "suc_khoe" });
  if (r && r.ok) hien("#2e9e57", `Máy chủ sẵn sàng: ${r.data.mo_hinh || "?"}${r.data.gpu ? " (" + r.data.gpu + ")" : ""}`);
  else hien("#c0392b", (r && r.loi) || "Không có trả lời.");
}

$("luu").addEventListener("click", async () => { await luu(); hien("#9aa7b4", "Đã lưu."); });
$("kiem").addEventListener("click", kiem);
nap();
