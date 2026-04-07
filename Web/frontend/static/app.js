const MODEL_LABELS = {
  resnet50: "ResNet50",
  efficientnet_b0: "EfficientNet-B0",
  vit_b_16: "ViT-B/16",
};

const REQUEST_CONTROLLERS = {
  overview: null,
  classify: null,
  check: null,
};

const DEFAULT_TOP_K = 3;
const MAX_TOP_K = 10;

async function requestJson(url, options = {}) {
  const res = await fetch(url, options);
  let payload;
  try {
    payload = await res.json();
  } catch (_err) {
    payload = { ok: false, error: "Phản hồi không phải JSON hợp lệ." };
  }
  if (!res.ok || !payload.ok) {
    const message = payload && payload.error ? payload.error : `HTTP ${res.status}`;
    throw new Error(message);
  }
  return payload;
}

function modelLabel(name) {
  return MODEL_LABELS[name] || name || "Không xác định";
}

function beginAbortableRequest(key) {
  const prev = REQUEST_CONTROLLERS[key];
  if (prev) {
    prev.abort();
  }
  const controller = new AbortController();
  REQUEST_CONTROLLERS[key] = controller;
  return controller;
}

function endAbortableRequest(key, controller) {
  if (REQUEST_CONTROLLERS[key] === controller) {
    REQUEST_CONTROLLERS[key] = null;
  }
}

function isAbortError(err) {
  return err && (err.name === "AbortError" || String(err).includes("AbortError"));
}

function setButtonBusy(button, busy, busyText) {
  if (!button) {
    return;
  }

  if (!button.dataset.originalText) {
    button.dataset.originalText = button.textContent || "";
  }

  button.disabled = busy;
  button.textContent = busy ? busyText : button.dataset.originalText;
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmtPct(value) {
  const n = toNumber(value);
  return n === null ? "n/a" : `${(n * 100).toFixed(2)}%`;
}

function fmtF1(value) {
  const n = toNumber(value);
  return n === null ? "n/a" : n.toFixed(4);
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function badge(label, cls) {
  return `<span class="badge ${cls}">${escapeHtml(label)}</span>`;
}

function asPre(obj) {
  return `<pre>${escapeHtml(JSON.stringify(obj, null, 2))}</pre>`;
}

function normalizeTopKInput(form) {
  const input = form.querySelector("input[name='top_k']");
  if (!input) {
    return DEFAULT_TOP_K;
  }

  const n = Number(input.value);
  const normalized = Number.isFinite(n)
    ? Math.max(1, Math.min(MAX_TOP_K, Math.trunc(n)))
    : DEFAULT_TOP_K;
  input.value = String(normalized);
  return normalized;
}

function renderOverview(data) {
  const panel = document.getElementById("overviewPanel");
  const dataset = data.dataset || {};
  const train = dataset.train || {};
  const val = dataset.val || {};
  const test = dataset.test || {};
  const ann = data.prescription_annotation_stats || {};
  const cache = data.cache || {};
  const generatedAt = cache.generated_at_unix
    ? new Date(Number(cache.generated_at_unix) * 1000).toLocaleString("vi-VN")
    : "n/a";
  const cacheTtl = Number(cache.overview_cache_ttl_sec || 0);

  const classEqual = dataset.class_sets_equal
    ? badge("Bộ class giữa train/val/test đang đồng nhất", "ok")
    : badge("Bộ class giữa train/val/test đang lệch", "warn");

  const modelRows = (data.models || [])
    .map((m) => {
      const state = m.checkpoint_exists
        ? badge("Checkpoint sẵn sàng", "ok")
        : badge("Thiếu checkpoint", "warn");
      const classCount = m.class_count ?? "n/a";
      return `<li><strong>${escapeHtml(modelLabel(m.model_name))}</strong> ${state} · số class=${escapeHtml(classCount)}</li>`;
    })
    .join("");

  const evalRows = (data.evaluation_summary || [])
    .map((r) => {
      const model = modelLabel(r.model);
      const acc = fmtPct(r.accuracy);
      const f1 = fmtF1(r.macro_f1);
      const samples = Number(r.num_samples || 0);
      return `<li>${escapeHtml(model)}: accuracy=${acc} · macro-F1=${f1} · samples=${samples}</li>`;
    })
    .join("");

  panel.innerHTML = `
    <div>${classEqual}</div>
    <ul class="data-list">
      <li><strong>Train:</strong> class=${train.class_count ?? 0}, ảnh=${train.total_images ?? 0}, lớp rỗng=${(train.empty_classes || []).length}</li>
      <li><strong>Val:</strong> class=${val.class_count ?? 0}, ảnh=${val.total_images ?? 0}, lớp rỗng=${(val.empty_classes || []).length}</li>
      <li><strong>Test:</strong> class=${test.class_count ?? 0}, ảnh=${test.total_images ?? 0}, lớp rỗng=${(test.empty_classes || []).length}</li>
      <li><strong>Prescription annotation:</strong> dòng=${ann.rows ?? 0}, class range=${ann.target_class_min ?? "n/a"}..${ann.target_class_max ?? "n/a"}</li>
      <li><strong>Overview cache:</strong> cập nhật lúc ${escapeHtml(generatedAt)} · TTL=${Number.isFinite(cacheTtl) ? cacheTtl : 0}s</li>
    </ul>
    <h3>Trạng thái mô hình</h3>
    <ul class="data-list">${modelRows || "<li>Chưa tìm thấy mô hình.</li>"}</ul>
    <h3>Tóm tắt đánh giá gần nhất</h3>
    <ul class="data-list">${evalRows || "<li>Chưa có file evaluation_summary.csv.</li>"}</ul>
  `;
}

async function loadOverview(forceRefresh = false) {
  const panel = document.getElementById("overviewPanel");
  panel.innerHTML = "Đang tải dữ liệu tổng quan...";

  const controller = beginAbortableRequest("overview");
  const endpoint = forceRefresh ? "/api/overview?force=1" : "/api/overview";

  try {
    const res = await requestJson(endpoint, { signal: controller.signal });
    renderOverview(res.data || {});
  } catch (err) {
    if (isAbortError(err)) {
      return;
    }
    panel.innerHTML = `<pre>Không thể tải tổng quan:\n${escapeHtml(String(err.message || err))}</pre>`;
  } finally {
    endAbortableRequest("overview", controller);
  }
}

async function handleClassifySubmit(event) {
  event.preventDefault();
  const output = document.getElementById("classifyResult");
  const form = event.currentTarget;
  const submitButton = form.querySelector("button[type='submit']");
  const fileInput = form.querySelector("input[name='image']");

  if (!fileInput || !fileInput.files || !fileInput.files.length) {
    output.innerHTML = "<pre>Vui lòng chọn 1 ảnh viên thuốc trước khi phân tích.</pre>";
    return;
  }

  normalizeTopKInput(form);
  output.textContent = "Đang chạy phân loại ảnh...";
  setButtonBusy(submitButton, true, "Đang phân tích...");

  const controller = beginAbortableRequest("classify");

  try {
    const formData = new FormData(form);
    const res = await requestJson("/api/classify", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    const d = res.data || {};
    const pred = d.predicted || {};
    const top = d.top_k || [];
    const topRows = top
      .map((x, i) => {
        const cid = x.class_id ?? "n/a";
        const cls = x.class_name || "n/a";
        const med = x.medicine_name || cls;
        return `<li>#${i + 1} ${escapeHtml(cls)} (id=${escapeHtml(cid)}) · độ tin cậy=${fmtPct(x.confidence)} · thuốc=${escapeHtml(med)}</li>`;
      })
      .join("");

    output.innerHTML = `
      <div>${badge("Phân loại thành công", "ok")}</div>
      <p><strong>Lớp dự đoán:</strong> ${escapeHtml(pred.class_name || "n/a")} (id=${escapeHtml(pred.class_id ?? "n/a")})</p>
      <p><strong>Tên thuốc:</strong> ${escapeHtml(pred.medicine_name || "n/a")}</p>
      <p><strong>Độ tin cậy:</strong> ${fmtPct(pred.confidence)}</p>
      <p><strong>Mô hình:</strong> ${escapeHtml(modelLabel(d.model_name))}</p>
      <p><strong>Checkpoint:</strong> ${escapeHtml(d.checkpoint || "n/a")}</p>
      <h3>Top-K</h3>
      <ul class="data-list">${topRows || "<li>Không có kết quả Top-K.</li>"}</ul>
      <details>
        <summary>Chi tiết JSON</summary>
        ${asPre(res)}
      </details>
    `;
  } catch (err) {
    if (isAbortError(err)) {
      output.textContent = "Đã hủy yêu cầu trước đó để xử lý yêu cầu mới nhất.";
      return;
    }
    output.innerHTML = `<pre>Phân loại thất bại:\n${escapeHtml(String(err.message || err))}</pre>`;
  } finally {
    endAbortableRequest("classify", controller);
    setButtonBusy(submitButton, false, "");
  }
}

async function handleCheckSubmit(event) {
  event.preventDefault();
  const output = document.getElementById("checkResult");
  const form = event.currentTarget;
  const submitButton = form.querySelector("button[type='submit']");
  const prescriptionInput = form.querySelector("input[name='prescription_image']");
  const pillInput = form.querySelector("input[name='pill_images']");

  if (!prescriptionInput || !prescriptionInput.files || !prescriptionInput.files.length) {
    output.innerHTML = "<pre>Vui lòng chọn ảnh toa thuốc/hóa đơn.</pre>";
    return;
  }

  const pillFiles = Array.from(pillInput?.files || []);
  if (!pillFiles.length) {
    output.innerHTML = "<pre>Vui lòng chọn ít nhất 1 ảnh viên thuốc.</pre>";
    return;
  }

  const maxFiles = Number(pillInput?.dataset.maxFiles || 30);
  if (pillFiles.length > maxFiles) {
    output.innerHTML = `<pre>Bạn đã chọn ${pillFiles.length} ảnh. Hệ thống chỉ hỗ trợ tối đa ${maxFiles} ảnh mỗi lần phân tích.</pre>`;
    return;
  }

  normalizeTopKInput(form);
  output.textContent = "Đang phân tích toa thuốc...";
  setButtonBusy(submitButton, true, "Đang phân tích...");

  const controller = beginAbortableRequest("check");

  try {
    const formData = new FormData(form);
    const res = await requestJson("/api/check-prescription", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    const d = res.data || {};
    const context = d.prescription_context || {};
    const items = d.items || [];

    let tfBadge = badge("Chưa xác định True/False", "warn");
    if (d.analysis_true_false === true) {
      tfBadge = badge("TRUE: Tất cả viên thuốc thuộc toa", "ok");
    } else if (d.analysis_true_false === false) {
      tfBadge = badge("FALSE: Có ít nhất một viên ngoài toa", "warn");
    }

    const contextBadge = context.found
      ? badge("Đã tìm thấy ngữ cảnh toa thuốc trong CSV", "ok")
      : badge("Không tìm thấy ngữ cảnh toa thuốc trong CSV", "warn");

    const itemRows = items
      .map((item) => {
        const pred = item.classification?.predicted || {};
        const status = item.is_in_prescription === true
          ? badge("Trong toa", "ok")
          : item.is_out_of_prescription === true
            ? badge("Ngoài toa", "warn")
            : badge("Chưa rõ", "warn");

        const conf = fmtPct(pred.confidence);
        const cls = pred.class_name || "n/a";
        const cid = pred.class_id ?? "n/a";
        const filename = item.pill_file?.original_name || "pill";

        return `<li>${status} <strong>${escapeHtml(filename)}</strong> → ${escapeHtml(cls)} (id=${escapeHtml(cid)}, độ tin cậy=${conf})</li>`;
      })
      .join("");

    const classesInPrescription = (context.classes_in_prescription || []).join(", ") || "n/a";
    const lookupBlock = d.annotation_lookup
      ? `
      <h3>Đối chiếu bổ sung từ annotation CSV</h3>
      ${asPre(d.annotation_lookup)}
      `
      : "";

    output.innerHTML = `
      <div>${tfBadge}</div>
      <div>${contextBadge}</div>
      <p><strong>Ảnh toa/hóa đơn:</strong> ${escapeHtml(d.prescription_file?.original_name || "n/a")}</p>
      <p><strong>Danh sách class trong toa:</strong> ${escapeHtml(classesInPrescription)}</p>
      <p><strong>Ghi chú khớp ngữ cảnh:</strong> ${escapeHtml(context.reason || "n/a")}</p>
      <h3>Kết quả theo từng viên thuốc</h3>
      <ul class="data-list">${itemRows || "<li>Không có ảnh viên thuốc để phân tích.</li>"}</ul>
      ${lookupBlock}
      <details>
        <summary>Chi tiết JSON</summary>
        ${asPre(res)}
      </details>
    `;
  } catch (err) {
    if (isAbortError(err)) {
      output.textContent = "Đã hủy yêu cầu trước đó để xử lý yêu cầu mới nhất.";
      return;
    }
    output.innerHTML = `<pre>Phân tích toa thuốc thất bại:\n${escapeHtml(String(err.message || err))}</pre>`;
  } finally {
    endAbortableRequest("check", controller);
    setButtonBusy(submitButton, false, "");
  }
}

function updateFilePreview(input) {
  const previewId = input.dataset.previewTarget;
  if (!previewId) {
    return;
  }

  const target = document.getElementById(previewId);
  if (!target) {
    return;
  }

  const files = Array.from(input.files || []);
  const maxFiles = Number(input.dataset.maxFiles || 0);
  if (!files.length) {
    if (input.name === "image") {
      target.textContent = "Chưa chọn ảnh.";
    } else if (input.name === "prescription_image") {
      target.textContent = "Chưa chọn ảnh toa/hóa đơn.";
    } else {
      target.textContent = "Chưa chọn ảnh viên thuốc.";
    }
    return;
  }

  if (input.name === "pill_images" && Number.isFinite(maxFiles) && maxFiles > 0 && files.length > maxFiles) {
    target.textContent = `Đã chọn ${files.length} ảnh (vượt quá giới hạn ${maxFiles}).`;
    return;
  }

  if (files.length === 1) {
    target.textContent = `Đã chọn: ${files[0].name}`;
    return;
  }

  const names = files.slice(0, 3).map((f) => f.name).join(", ");
  const remain = files.length - 3;
  target.textContent = remain > 0
    ? `Đã chọn ${files.length} ảnh: ${names}, ...`
    : `Đã chọn ${files.length} ảnh: ${names}`;
}

function setupEvents() {
  const refreshButton = document.getElementById("refreshOverview");
  const classifyForm = document.getElementById("classifyForm");
  const checkForm = document.getElementById("checkForm");

  refreshButton?.addEventListener("click", () => loadOverview(true));
  classifyForm?.addEventListener("submit", handleClassifySubmit);
  checkForm?.addEventListener("submit", handleCheckSubmit);

  const inputs = document.querySelectorAll("input[type='file'][data-preview-target]");
  inputs.forEach((input) => {
    input.addEventListener("change", () => updateFilePreview(input));
  });
}

window.addEventListener("DOMContentLoaded", () => {
  setupEvents();
  loadOverview();
});
