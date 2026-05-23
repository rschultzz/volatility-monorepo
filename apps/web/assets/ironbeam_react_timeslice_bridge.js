(function () {
  function setDashInputValue(input, value) {
    var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    nativeSetter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function postParentTimesToIframe() {
    var input = document.getElementById("ib-react-timeslice-parent");
    var frame = document.getElementById("ib-react-preview-frame");
    if (!input || !frame || !frame.contentWindow) return;

    var raw = input.value || "";
    if (!raw) raw = JSON.stringify({ times: [] });

    try {
      var payload = JSON.parse(raw);
      frame.contentWindow.postMessage(
        {
          type: "ib-parent-timeslices",
          times: Array.isArray(payload.times) ? payload.times : [],
        },
        "*"
      );
    } catch (err) {
      console.error("ib-react-timeslice-parent post error", err);
    }
  }

  window.addEventListener("message", function (event) {
    var data = event && event.data;
    if (!data || data.type !== "ib-react-timeslices") return;

    var input = document.getElementById("ib-react-timeslice-bridge");
    if (!input) return;

    try {
      var payload = JSON.stringify({ times: Array.isArray(data.times) ? data.times : [] });
      setDashInputValue(input, payload);
    } catch (err) {
      console.error("ib-react-timeslice-bridge error", err);
    }
  });

  var lastParentValue = null;
  function watchParentValue() {
    var input = document.getElementById("ib-react-timeslice-parent");
    if (!input) return;
    var current = input.value || "";
    if (current !== lastParentValue) {
      lastParentValue = current;
      postParentTimesToIframe();
    }
  }

  window.addEventListener("load", function () {
    var frame = document.getElementById("ib-react-preview-frame");
    if (frame) {
      frame.addEventListener("load", function () {
        setTimeout(postParentTimesToIframe, 50);
      });
    }
    setTimeout(postParentTimesToIframe, 150);
    setInterval(watchParentValue, 250);
  });
})();
