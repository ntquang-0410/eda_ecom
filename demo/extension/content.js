// Content script trang chi tiết 1688: lấy tên sản phẩm + bảng thuộc tính, gửi máy chủ dịch, thay chữ tại chỗ.
// Chỉ dịch hai thứ mô hình đã học (title, thuộc tính); không đụng ảnh mô tả, đánh giá, tên phân loại SKU, giao diện.
// Bộ chọn đo trên 5 trang thật (thời trang, điện tử, giày, túi, thực phẩm): số dòng đọc từ DOM khớp số thuộc tính trong JSON nhúng.
(() => {
  if (window.__dich1688) return;
  window.__dich1688 = true;

  const SEL = {
    title: ".module-od-title h1",
    cpv: ".cpv-decision-attr .cpv-attr-item",          // khối thuộc tính chính: nhãn + giá trị là hai span
    cpv_ten: ".cpv-attr-label",
    cpv_gt: ".cpv-attr-value",
    bang: ".attr-descriptions tr",                      // bảng thuộc tính: th nhãn, td giá trị, mỗi hàng 1-2 cặp
    bang_ten: "th.ant-descriptions-item-label",
  };
  const CO_HAN = /[一-鿿]/;
  const KHOI = 20;                                      // số dòng thuộc tính mỗi lượt gọi

  let trang_thai = "goc";                               // goc | can_zh | dang_dich | vi | goc_sau_dich
  let tru = null;                                       // {tieu_de, hang: [...]} các phần tử đã dịch
  let da_thu_ep = false;                                // đã thử ép trang sang tiếng Trung (sau khi tải lại)
  const CO_NHO = "dich1688_tu_dong";

  // ---------- giao diện nút nổi (shadow DOM để không dính CSS của 1688) ----------
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;right:16px;bottom:96px;z-index:2147483647;";
  const sh = host.attachShadow({ mode: "open" });
  sh.innerHTML = `
    <style>
      .hop { font: 13px/1.4 system-ui, "Segoe UI", sans-serif; background: #fff; border: 1px solid #1f6f5c; border-radius: 10px;
             box-shadow: 0 4px 16px rgba(0,0,0,.25); padding: 8px; width: 220px; color: #1d2733; }
      button { width: 100%; padding: 8px 10px; border: 0; border-radius: 7px; background: #1f6f5c; color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
      button:disabled { background: #7a8c86; cursor: default; }
      .tt { margin-top: 6px; min-height: 1.4em; font-size: 12px; color: #4a5866; word-break: break-word; }
      .tt.loi { color: #b3261e; }
    </style>
    <div class="hop"><button id="nut">Dịch sang tiếng Việt</button><div class="tt" id="tt"></div></div>`;
  const nut = sh.getElementById("nut");
  const tt = sh.getElementById("tt");
  const bao = (chu, loi = false) => { tt.textContent = chu; tt.classList.toggle("loi", loi); };
  document.documentElement.appendChild(host);

  // ---------- đọc DOM ----------
  const chu_cua = (el) => (el ? el.textContent.replace(/\s+/g, " ").trim() : "");

  function lay_title() {
    const el = document.querySelector(SEL.title);
    return el ? { el, zh: chu_cua(el) } : null;
  }

  function lay_thuoc_tinh() {
    const hang = [];
    document.querySelectorAll(SEL.cpv).forEach((it) => {
      const a = it.querySelector(SEL.cpv_ten), b = it.querySelector(SEL.cpv_gt);
      if (a && b) hang.push({ ten_el: a, gt_el: b });
    });
    document.querySelectorAll(SEL.bang).forEach((tr) => {
      tr.querySelectorAll(SEL.bang_ten).forEach((th) => {
        const td = th.nextElementSibling;
        if (!td) return;
        hang.push({ ten_el: th.querySelector("span") || th, gt_el: td.querySelector(".field-value") || td });
      });
    });
    hang.forEach((h) => { h.ten = chu_cua(h.ten_el); h.gt = chu_cua(h.gt_el); });
    return hang.filter((h) => h.ten && h.gt && (CO_HAN.test(h.ten) || CO_HAN.test(h.gt)));
  }

  async function cho_title(ms) {
    const t0 = Date.now();
    let ti = lay_title();
    while (!ti && Date.now() - t0 < ms) {
      await new Promise((r) => setTimeout(r, 500));
      ti = lay_title();
    }
    return ti;
  }

  async function cho_thuoc_tinh(ms) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      const h = lay_thuoc_tinh();
      if (h.length) return h;
      window.scrollBy(0, 600);                          // khối thuộc tính có thể tải khi cuộn tới
      await new Promise((r) => setTimeout(r, 700));
    }
    return [];
  }

  // ---------- thay chữ tại chỗ, giữ bản gốc ----------
  function dat_chu(el, t) {
    if (el.dataset.goc === undefined) el.dataset.goc = chu_cua(el);
    const con = el.childNodes;
    if (con.length === 1 && con[0].nodeType === 3) con[0].nodeValue = t;
    else el.textContent = t;
  }
  function xem_goc(el) { el.title = "Bản gốc: " + el.dataset.goc; }
  function tra_goc(el) {
    if (el.dataset.goc === undefined) return;
    const con = el.childNodes;
    if (con.length === 1 && con[0].nodeType === 3) con[0].nodeValue = el.dataset.goc;
    else el.textContent = el.dataset.goc;
    el.removeAttribute("title");
  }

  // ---------- gọi máy chủ qua background ----------
  async function goi(body) {
    let r;
    try { r = await chrome.runtime.sendMessage({ loai_yeu_cau: "dich", body }); }
    catch (e) { throw new Error("Extension bị nạp lại, hãy tải lại trang."); }
    if (!r || !r.ok) throw new Error((r && r.loi) || "Không có trả lời từ máy chủ.");
    return r.data;
  }

  // ---------- luồng dịch ----------
  async function dich() {
    trang_thai = "dang_dich";
    nut.disabled = true;
    try {
      const ti = await cho_title(da_thu_ep ? 10000 : 0);
      if (!ti) {
        bao("Không tìm thấy tên sản phẩm trên trang này.", true);
        trang_thai = "goc";
        return;
      }
      if (!CO_HAN.test(ti.zh)) {
        if (!da_thu_ep) {                               // trang đang là bản 1688 tự dịch (Anh/Việt): đề nghị chuyển sang bản tiếng Trung
          trang_thai = "can_zh";
          nut.textContent = "Chuyển trang sang tiếng Trung";
          bao("Trang đang hiện bản 1688 tự dịch (không có chữ Hán). Bấm nút để chuyển sang bản tiếng Trung, trang sẽ tải lại và tự dịch.");
        } else {
          trang_thai = "goc";
          bao("1688 vẫn trả trang không phải tiếng Trung. Đổi ô ngôn ngữ trên đầu trang sang 中文 (hoặc mở bằng cửa sổ ẩn danh) rồi bấm lại.", true);
        }
        return;
      }
      tru = { tieu_de: ti, hang: [] };
      bao("Đang dịch tên sản phẩm…");
      const rt = await goi({ loai: "title", zh: ti.zh });
      dat_chu(ti.el, rt.vi); xem_goc(ti.el);

      bao("Đang tìm bảng thuộc tính…");
      const hang = await cho_thuoc_tinh(8000);
      if (!hang.length) bao("Đã dịch tên. Không thấy bảng thuộc tính có chữ Hán.");
      let xong = 0, khong_tach = 0, lech = 0;
      for (let i = 0; i < hang.length; i += KHOI) {
        const lo = hang.slice(i, i + KHOI);
        bao(`Đang dịch thuộc tính ${Math.min(i + KHOI, hang.length)}/${hang.length}…`);
        const k = await goi({ loai: "khoi", title: ti.zh, cap: lo.map((h) => [h.ten, h.gt]) });
        if (k.lech_dong) lech++;
        if (!Array.isArray(k.cap) || k.cap.length !== lo.length) throw new Error("Máy chủ trả sai số dòng.");
        lo.forEach((h, j) => {
          const [t_vi, g_vi] = k.cap[j];
          if (g_vi === null || g_vi === undefined || !t_vi) { khong_tach++; return; }   // dòng không tách được: giữ nguyên gốc
          dat_chu(h.ten_el, t_vi); xem_goc(h.ten_el);
          dat_chu(h.gt_el, g_vi); xem_goc(h.gt_el);
          tru.hang.push(h);
          xong++;
        });
      }
      trang_thai = "vi";
      nut.textContent = "Xem bản gốc (tiếng Trung)";
      bao(`Xong: tên + ${xong} thuộc tính${khong_tach ? `, ${khong_tach} dòng giữ nguyên` : ""}${lech ? ` (${lech} khối dịch lại từng dòng)` : ""}.`);
    } catch (e) {
      bao(e.message, true);
      trang_thai = tru && (tru.hang.length || (tru.tieu_de && tru.tieu_de.el.dataset.goc !== undefined)) ? "vi" : "goc";
      if (trang_thai === "vi") nut.textContent = "Xem bản gốc (tiếng Trung)";
    } finally {
      nut.disabled = false;
    }
  }

  function xem_ban_goc() {
    if (!tru) return;
    [tru.tieu_de.el, ...tru.hang.flatMap((h) => [h.ten_el, h.gt_el])].forEach((el) => {
      el.dataset.vi = chu_cua(el);
      tra_goc(el);
    });
    trang_thai = "goc_sau_dich";
    nut.textContent = "Xem bản dịch tiếng Việt";
    bao("Đang hiển thị bản gốc.");
  }

  function xem_ban_dich() {
    [tru.tieu_de.el, ...tru.hang.flatMap((h) => [h.ten_el, h.gt_el])].forEach((el) => {
      if (el.dataset.vi === undefined) return;
      const goc = el.dataset.goc;
      dat_chu(el, el.dataset.vi);
      el.dataset.goc = goc;
      xem_goc(el);
    });
    trang_thai = "vi";
    nut.textContent = "Xem bản gốc (tiếng Trung)";
    bao("Đang hiển thị bản dịch tiếng Việt.");
  }

  async function ep_tieng_trung() {
    nut.disabled = true;
    bao("Đang chuyển sang tiếng Trung…");
    let r;
    try { r = await chrome.runtime.sendMessage({ loai_yeu_cau: "ep_zh" }); }
    catch (e) { r = { ok: false, loi: "Extension bị nạp lại, hãy tải lại trang." }; }
    if (!r || !r.ok) {
      bao((r && r.loi) || "Không chuyển được ngôn ngữ.", true);
      nut.disabled = false;
      return;
    }
    try { sessionStorage.setItem(CO_NHO, "1"); } catch (e) { /* không có thì người dùng bấm tay sau khi tải lại */ }
    location.reload();
  }

  nut.addEventListener("click", () => {
    if (trang_thai === "can_zh") ep_tieng_trung();
    else if (trang_thai === "goc") dich();
    else if (trang_thai === "vi") xem_ban_goc();
    else if (trang_thai === "goc_sau_dich") xem_ban_dich();
  });

  // Sau khi tự chuyển sang tiếng Trung và tải lại: tự bấm dịch một lần.
  try {
    if (sessionStorage.getItem(CO_NHO) === "1") {
      sessionStorage.removeItem(CO_NHO);
      da_thu_ep = true;
      setTimeout(() => { if (trang_thai === "goc") dich(); }, 1000);
    }
  } catch (e) { /* bỏ qua */ }
})();
