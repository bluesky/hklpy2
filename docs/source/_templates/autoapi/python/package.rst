{% extends "python/module.rst" %}
{% block content %}
{% if obj.name == "hklpy2" %}

.. figure:: /_static/hklpy2-overview.svg
   :alt: hklpy2 package architecture overview
   :align: center

   Overview of the major sections of |hklpy2|.
   See :ref:`overview.architecture` for detailed diagrams of each section.

{% endif %}
{{ super() }}
{% endblock %}
