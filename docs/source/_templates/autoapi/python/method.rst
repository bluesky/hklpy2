{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id | length }}

   {% endif %}
.. py:method:: {% if is_own_page %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}{% if obj.type_params %}[{{ obj.type_params }}]{% endif %}({{ obj.args | shorten_type }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation | shorten_type }}{% endif %}
   {% for (args, return_annotation) in obj.overloads %}

               {%+ if is_own_page %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}({{ args | shorten_type }}){% if return_annotation is not none %} -> {{ return_annotation | shorten_type }}{% endif %}
   {% endfor %}
   {% for property in obj.properties %}

   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
