---
layout: default
permalink: /blog/
title: blog
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

<div class="post blog-index">
  <header class="post-header">
    <h1 class="post-title">{{ site.blog_name }}</h1>
    <p class="post-description">{{ site.blog_description }}</p>
  </header>

{% assign all_tags = site.tags | sort %}

  <nav class="ed-chips">
    <a class="on" href="{{ '/blog/' | relative_url }}">all</a>
    {% for tag in all_tags %}
      <a href="{{ tag[0] | slugify | prepend: '/blog/tag/' | relative_url }}">{{ tag[0] }}</a>
    {% endfor %}
  </nav>

{% if page.pagination.enabled %}
{% assign postlist = paginator.posts %}
{% else %}
{% assign postlist = site.posts %}
{% endif %}

{% assign current_year = '' %}
{% for post in postlist %}
{% assign year = post.date | date: '%Y' %}
{% if year != current_year %}
{% assign current_year = year %}

  <div class="ed-year">{{ year }}</div>
  {% endif %}
  {% if post.external_source == blank %}
    {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
  {% else %}
    {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
  {% endif %}
  {% if post.redirect == blank %}
    {% assign post_url = post.url | relative_url %}
  {% elsif post.redirect contains '://' %}
    {% assign post_url = post.redirect %}
  {% else %}
    {% assign post_url = post.redirect | relative_url %}
  {% endif %}
  <article class="ed-post">
    <span class="ed-date">{{ post.date | date: '%b %d, %Y' }} · {{ read_time }} min read</span>
    <h3><a href="{{ post_url }}">{{ post.title }}</a></h3>
    {% if post.description %}<p>{{ post.description }}</p>{% endif %}
    {% if post.tags.size > 0 %}
      <div class="ed-tags">
        {% for tag in post.tags %}
          <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">#{{ tag }}</a>
        {% endfor %}
      </div>
    {% endif %}
  </article>
{% endfor %}

{% if page.pagination.enabled %}
{% include pagination.liquid %}
{% endif %}

</div>
