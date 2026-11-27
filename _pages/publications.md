---
layout: page
permalink: /publications/
title: publications
description: "* equal contribution · ★ selected"
nav: true
nav_order: 2
scholar_profile_metrics: true
---

<!-- _pages/publications.md -->

{% if page.scholar_profile_metrics %}
{% include scholar_profile_metrics.liquid %}
{% endif %}

<div class="pub-controls">
<span>Sort:</span>
<button type="button" data-sort="year" class="on">by year</button>
<button type="button" data-sort="cites">by citations</button>
<span class="sep">|</span>
<button type="button" data-thumbs class="on">thumbnails</button>
</div>

<div class="publications" id="pub-list">

{% bibliography %}

</div>

<script>
  (function () {
    var root = document.getElementById("pub-list");
    if (!root) return;
    // Remember where every entry lives so "by year" can restore it exactly.
    var items = Array.prototype.map.call(root.querySelectorAll("ol.bibliography > li"), function (li, i) {
      var n = li.querySelector(".cite-count .n");
      return { li: li, parent: li.parentNode, order: i, cites: n ? parseInt(n.textContent, 10) || 0 : -1 };
    });
    var flat = document.createElement("ol");
    flat.className = "bibliography by-cites";
    root.appendChild(flat);
    flat.hidden = true;

    function show(mode) {
      var byYear = mode === "year";
      root.querySelectorAll("h2.bibliography, ol.bibliography:not(.by-cites)").forEach(function (el) {
        el.hidden = !byYear;
      });
      flat.hidden = byYear;
      var list = items.slice().sort(function (a, b) {
        return byYear ? a.order - b.order : b.cites - a.cites || a.order - b.order;
      });
      list.forEach(function (it) {
        (byYear ? it.parent : flat).appendChild(it.li);
      });
    }

    document.querySelectorAll(".pub-controls [data-sort]").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll(".pub-controls [data-sort]").forEach(function (x) {
          x.classList.toggle("on", x === b);
        });
        show(b.dataset.sort);
      });
    });
    var tb = document.querySelector(".pub-controls [data-thumbs]");
    tb.addEventListener("click", function () {
      tb.classList.toggle("on");
      root.classList.toggle("no-thumbs");
    });
  })();
</script>
