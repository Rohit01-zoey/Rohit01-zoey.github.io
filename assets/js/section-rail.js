// Home page section line: a hollow circle per section in the left margin.
// The section you're reading gets a filled circle and a bold label.
// Clicking a label scrolls that section to a little above the middle of the
// screen (not flush with the top), which is easier to read.
(function () {
  var rail = document.querySelector(".section-rail ol");
  var page = document.querySelector(".editorial-home");
  if (!rail || !page) return;

  // Where a clicked section lands, and the line a section must pass to count
  // as "current" (both as a fraction of the window height from the top).
  var LAND_AT = 0.3;
  var ACTIVE_AT = 0.42;
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function slug(text) {
    return (
      "sec-" +
      text
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "")
    );
  }

  var targets = [];
  var title = page.querySelector(".post-title");
  if (title) targets.push({ el: title, label: "About" });
  page.querySelectorAll("h2.section-label").forEach(function (h) {
    var span = h.querySelector("span");
    var label = (span || h).textContent.trim();
    if (label) targets.push({ el: h, label: label.charAt(0).toUpperCase() + label.slice(1) });
  });
  if (targets.length < 2) return;

  targets.forEach(function (t) {
    if (!t.el.id) t.el.id = slug(t.label);
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = "#" + t.el.id;
    a.textContent = t.label;
    a.addEventListener("click", function (e) {
      e.preventDefault();
      var y = t.el.getBoundingClientRect().top + window.scrollY - window.innerHeight * LAND_AT;
      window.scrollTo({ top: Math.max(0, y), behavior: reduceMotion ? "auto" : "smooth" });
      history.replaceState(null, "", "#" + t.el.id);
    });
    li.appendChild(a);
    rail.appendChild(li);
    t.li = li;
  });

  function spy() {
    var line = window.innerHeight * ACTIVE_AT;
    var current = targets[0];
    targets.forEach(function (t) {
      if (t.el.getBoundingClientRect().top < line) current = t;
    });
    // At the very bottom of the page, light up the last section.
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) current = targets[targets.length - 1];
    targets.forEach(function (t) {
      t.li.classList.toggle("on", t === current);
      if (t === current) t.li.querySelector("a").setAttribute("aria-current", "true");
      else t.li.querySelector("a").removeAttribute("aria-current");
    });
  }

  window.addEventListener("scroll", spy, { passive: true });
  window.addEventListener("resize", spy);
  spy();
  document.querySelector(".section-rail").classList.add("ready");
})();
