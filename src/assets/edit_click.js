// Edit mode of src/app.py: a click ANYWHERE inside a panel (not only on a glass) is sent to Dash as data
// coordinates of that panel, in the store "edit-click". Plotly's own click event fires only on points; but when
// the mouse is released without dragging, Plotly re-dispatches a DOM "click", which is caught here.
// Python ignores the store unless "edit mode" is checked. A double click (Plotly's reset zoom) is not sent.
(function () {
  let timer = null;
  let last = 0, lastTrusted = null;
  document.addEventListener("click", function (e) {
    const host = document.getElementById("grid");
    const gd = host && host.querySelector(".js-plotly-plot");
    if (!gd || !gd._fullLayout || !gd.contains(e.target) || e.target.closest(".modebar")) return;
    const edit = document.querySelector("#edit-mode input[type=checkbox]");
    if (!edit || !edit.checked) return;

    const rect = gd.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    let hit = null;
    for (const sp of Object.values(gd._fullLayout._plots || {})) {
      const xa = sp.xaxis, ya = sp.yaxis;
      if (px >= xa._offset && px <= xa._offset + xa._length && py >= ya._offset && py <= ya._offset + ya._length) {
        hit = {xaxis: xa._id, x: xa.p2l(px - xa._offset), y: ya.p2l(py - ya._offset)};
        break;
      }
    }
    if (!hit) return;

    // one physical click usually arrives twice: the browser's own (trusted) and Plotly's re-dispatched copy
    // (~30 ms later). A pair of different kinds close together is one click; two of the same kind are a double click.
    const now = Date.now(), dt = now - last;
    if (dt < 150 && e.isTrusted !== lastTrusted) return;          // the copy of the click just handled
    last = now;
    lastTrusted = e.isTrusted;
    if (dt < 400) { clearTimeout(timer); timer = null; return; }  // double click: reset zoom, not a target
    clearTimeout(timer);
    timer = setTimeout(function () {
      window.dash_clientside.set_props("edit-click", {data: Object.assign(hit, {t: now})});
    }, 400);
  }, true);
})();
