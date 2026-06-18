function initGlossaryTerms() {
  const glossaryDataEl = document.getElementById("glossary-data");
  if (!glossaryDataEl) {
    return;
  }

  let glossaryBySlug = {};
  try {
    const glossaryData = JSON.parse(glossaryDataEl.textContent);
    glossaryBySlug = Object.fromEntries(glossaryData.map((entry) => [entry.slug, entry]));
  } catch (error) {
    console.warn("Failed to parse glossary data.", error);
    return;
  }

  $(".glossary-term[data-glossary-slug]").each(function () {
    const $term = $(this);
    const slug = $term.data("glossary-slug");
    const entry = glossaryBySlug[slug];
    if (!entry) {
      return;
    }

    if ($term.data("bs.popover")) {
      $term.popover("dispose");
    }

    const content =
      entry.definitionHtml +
      `<p class="glossary-popover-footer mb-0"><a class="glossary-popover-link" href="${entry.url}">View in glossary &rarr;</a></p>`;

    $term.popover({
      trigger: "hover focus",
      html: true,
      container: "body",
      content,
      title: entry.term,
    });

    $term.off("click.glossary").on("click.glossary", function (event) {
      event.preventDefault();
    });
  });
}

function scheduleGlossaryInit() {
  initGlossaryTerms();
}

$(document).ready(scheduleGlossaryInit);
$(window).on("load", scheduleGlossaryInit);
