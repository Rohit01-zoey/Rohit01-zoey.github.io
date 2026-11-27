# Cache-busting query string for the compiled stylesheet.
#
# The jekyll-cache-bust gem's `bust_css_cache` hashes `assets/_sass`, which does
# not exist in this site (partials live in `_sass/`), so it always produced the
# same `?v=` value and browsers kept serving stale CSS after a deploy. This
# filter hashes the real Sass sources plus the main stylesheet instead.
require "digest/md5"

module Jekyll
  module SassCacheBust
    def bust_sass_cache(file_name)
      files = Dir[File.join("_sass", "**", "*.scss")].sort
      files << File.join("assets", "css", "main.scss")
      contents = files.select { |f| File.file?(f) }.map { |f| File.read(f) }.join
      "#{file_name}?v=#{Digest::MD5.hexdigest(contents)}"
    end
  end
end

Liquid::Template.register_filter(Jekyll::SassCacheBust)
