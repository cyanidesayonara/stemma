/* The theme toggle, like the sun button in the app's corner: dark or light,
   remembered for next time, otherwise the system's choice. */
(function () {
  "use strict";

  var root = document.documentElement;
  var button = document.querySelector("button.theme");
  var logo = document.querySelector("img.logo");
  var source = document.querySelector(".mark source");
  // the script picks the logo from here on, so the <picture> fallback for
  // visitors without JavaScript steps aside
  if (source) { source.remove(); }
  var system = window.matchMedia("(prefers-color-scheme: light)");

  function current() {
    var picked = root.getAttribute("data-theme");
    if (picked) { return picked; }
    return system.matches ? "light" : "dark";
  }

  function show() {
    var theme = current();
    if (logo) { logo.src = logo.getAttribute("data-" + theme); }
    if (button) {
      var other = theme === "dark" ? "light" : "dark";
      button.setAttribute("aria-label", "Switch to the " + other + " theme");
    }
    // the browser bar follows the theme shown, picked or the system's: the
    // page background, read from the palette
    var bar = getComputedStyle(root).getPropertyValue("--mantle").trim();
    if (bar) {
      document.querySelectorAll('meta[name="theme-color"]').forEach(
        function (meta) { meta.setAttribute("content", bar); });
    }
    document.dispatchEvent(new CustomEvent("stemma-theme"));
  }

  if (button) {
    button.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("stemma-theme", next); } catch (e) {}
      show();
    });
  }
  if (system.addEventListener) { system.addEventListener("change", show); }
  show();
})();

/* The feature drawings animate only while their card is on screen. */
(function () {
  "use strict";

  var cards = document.querySelectorAll("ul.features li");
  if (!("IntersectionObserver" in window)) {
    cards.forEach(function (card) { card.classList.add("live"); });
    return;
  }
  var seen = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      entry.target.classList.toggle("live", entry.isIntersecting);
    });
  });
  cards.forEach(function (card) { seen.observe(card); });
})();
