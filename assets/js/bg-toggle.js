// Home page "Background" toggle.
//   More: the short summary gives way and the full text is typed out in place.
//   Less: the words of the full text squeeze together and vanish, then the
//         short summary un-merges into view.
// Markup: #bg-toggle > .bg-short, .bg-full[hidden], button.bg-btn
(function () {
  var root = document.getElementById("bg-toggle");
  if (!root) return;
  var shortEl = root.querySelector(".bg-short");
  var fullEl = root.querySelector(".bg-full");
  var btn = root.querySelector(".bg-btn");
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var busy = false;
  var expanded = false;

  var BLOCKS = "P,LI,UL,OL";
  var TYPE_MS = 1800; // total time to type the full background, regardless of frame rate

  function textNodes(el) {
    var out = [];
    var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        return /\S/.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });
    while (walker.nextNode()) out.push(walker.currentNode);
    return out;
  }

  function blockAncestors(node) {
    var list = [];
    for (var el = node.parentNode; el && el !== fullEl; el = el.parentNode) {
      if (BLOCKS.indexOf(el.nodeName) > -1) list.push(el);
    }
    return list;
  }

  function setLabel() {
    btn.innerHTML = expanded ? "Less &uarr;" : "More &darr;";
    btn.setAttribute("aria-expanded", expanded ? "true" : "false");
  }

  // Wrap every word (with its trailing space) in a span so it can be squeezed.
  function wrapWords(el) {
    var spans = [];
    var originals = [];
    textNodes(el).forEach(function (tn) {
      var frag = document.createDocumentFragment();
      var parts = tn.nodeValue.match(/\s+|\S+\s*/g) || [];
      parts.forEach(function (part) {
        if (/^\s+$/.test(part)) {
          frag.appendChild(document.createTextNode(part));
        } else {
          var s = document.createElement("span");
          s.className = "w";
          s.textContent = part;
          frag.appendChild(s);
          spans.push(s);
        }
      });
      originals.push({ node: tn, frag: frag });
    });
    originals.forEach(function (o) {
      o.marker = o.frag.firstChild;
      o.count = o.frag.childNodes.length;
      o.node.parentNode.replaceChild(o.frag, o.node);
    });
    return {
      spans: spans,
      restore: function () {
        originals.forEach(function (o) {
          var first = o.marker;
          var parent = first.parentNode;
          parent.insertBefore(o.node, first);
          for (var i = 0; i < o.count; i++) {
            var next = o.node.nextSibling;
            parent.removeChild(next);
          }
        });
      },
    };
  }

  function freezeWidths(spans) {
    spans.forEach(function (s) {
      s.style.maxWidth = s.getBoundingClientRect().width + 1 + "px";
    });
  }

  function staggered(spans, total, fn, done) {
    var n = spans.length;
    if (!n) return done();
    var step = total / n;
    spans.forEach(function (s, i) {
      setTimeout(function () {
        fn(s);
      }, i * step);
    });
    setTimeout(done, total + 450);
  }

  // ---------- More: type the full text ----------
  function expand() {
    busy = true;
    shortEl.classList.add("fade");
    setTimeout(function () {
      shortEl.hidden = true;
      shortEl.classList.remove("fade");

      var nodes = textNodes(fullEl).map(function (tn) {
        return { node: tn, text: tn.nodeValue, blocks: blockAncestors(tn) };
      });
      nodes.forEach(function (n) {
        n.node.nodeValue = "";
        n.blocks.forEach(function (b) {
          b.classList.add("pending");
        });
      });
      fullEl.hidden = false;

      var caret = document.createElement("span");
      caret.className = "caret";
      var total = nodes.reduce(function (sum, n) {
        return sum + n.text.length;
      }, 0);
      var typed = 0;
      var i = 0;
      var pos = 0;
      var t0 = performance.now();

      function frame(now) {
        // Time-based, so throttled/background tabs still finish on schedule.
        var budget = Math.min(total, Math.ceil(((now - t0) / TYPE_MS) * total)) - typed;
        while (budget > 0 && i < nodes.length) {
          var n = nodes[i];
          if (pos === 0) {
            n.blocks.forEach(function (b) {
              b.classList.remove("pending");
            });
          }
          var take = Math.min(budget, n.text.length - pos);
          pos += take;
          budget -= take;
          typed += take;
          n.node.nodeValue = n.text.slice(0, pos);
          if (n.node.nextSibling !== caret) n.node.parentNode.insertBefore(caret, n.node.nextSibling);
          if (pos >= n.text.length) {
            i++;
            pos = 0;
          }
        }
        if (i < nodes.length) {
          requestAnimationFrame(frame);
        } else {
          setTimeout(function () {
            caret.remove();
          }, 600);
          expanded = true;
          setLabel();
          busy = false;
        }
      }
      requestAnimationFrame(frame);
    }, 180);
  }

  // ---------- Less: merge the words away, then un-merge the summary ----------
  function collapse() {
    busy = true;
    var start = root.getBoundingClientRect().top;
    var full = wrapWords(fullEl);
    freezeWidths(full.spans);
    void fullEl.offsetWidth; // commit widths before animating

    // Squeeze from the end of the text back toward the start.
    staggered(
      full.spans.slice().reverse(),
      650,
      function (s) {
        s.classList.add("m");
      },
      function () {
        fullEl.hidden = true;
        full.restore();

        var brief = wrapWords(shortEl);
        shortEl.hidden = false;
        freezeWidths(brief.spans);
        brief.spans.forEach(function (s) {
          s.dataset.w = s.style.maxWidth;
          s.classList.add("m");
        });
        void shortEl.offsetWidth;

        // Keep the section in view if the page got shorter above the fold.
        var now = root.getBoundingClientRect().top;
        if (now < 0 && start >= 0) root.scrollIntoView({ block: "start", behavior: "smooth" });

        staggered(
          brief.spans,
          380,
          function (s) {
            s.classList.remove("m");
          },
          function () {
            brief.restore();
            expanded = false;
            setLabel();
            busy = false;
          },
        );
      },
    );
  }

  btn.addEventListener("click", function () {
    if (busy) return;
    if (reduceMotion) {
      expanded = !expanded;
      shortEl.hidden = expanded;
      fullEl.hidden = !expanded;
      setLabel();
      return;
    }
    expanded ? collapse() : expand();
  });
})();
