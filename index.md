---
layout: page
title: Some dirty fieldwork lessons that might help others!!
tagline: Place for a tagline
---
{% include JB/setup %}

Before anything else this blog is powered by Jekyll. Want to know more?  [Jekyll Quick Start](http://jekyllbootstrap.com/usage/jekyll-quick-start.html)

Complete usage and documentation available at: [Jekyll Bootstrap](http://jekyllbootstrap.com)

## How it all started

Through some online courses
    
    Sometime in my first year I received a mail in my gmail about three online courses! Machine Learning, HCI and Databases. 
    For some unknown reason I decided to experiment with ML and HCI! Registered for the two courses.
    Since then it has been a lot of fun!
    
    Currently I am a research assistant at the CNRS lab with the greta team synthesizing laughter as part of the ILHAIRE project. 

The theme should reference these variables whenever needed.
    
## Sample Posts

This blog contains sample posts which help stage pages and blog data.
When you don't need the samples anymore just delete the `_posts/core-samples` folder.

    $ rm -rf _posts/core-samples

Here's a sample "posts list".

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>

## To-Do

This theme is still unfinished. If you'd like to be added as a contributor, [please fork](http://github.com/plusjade/jekyll-bootstrap)!
We need to clean up the themes, make theme usage guides with theme-specific markup examples.


