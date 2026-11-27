---
layout: page
title: projects
permalink: /projects/
description: Longer-running pieces of research and code.
nav: true
nav_order: 3
display_categories: [research, fun]
horizontal: false
---

<!-- pages/projects.md -->
<div class="ed-projects">
{% assign sorted_projects = site.projects | sort: "importance" %}
{% for project in sorted_projects %}
{% if project.redirect %}{% assign purl = project.redirect %}{% else %}{% assign purl = project.url | relative_url %}{% endif %}
  <a class="ed-proj" href="{{ purl }}">
    {% if project.img %}<img src="{{ project.img | relative_url }}" alt="{{ project.title }}">{% else %}<span class="ph"></span>{% endif %}
    <div>
      {% if project.category %}<span class="ed-date">{{ project.category | capitalize }}</span>{% endif %}
      <h3>{{ project.title }}</h3>
      <p>{{ project.description }}</p>
    </div>
  </a>
{% endfor %}
</div>
