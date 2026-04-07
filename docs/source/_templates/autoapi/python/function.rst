{% if obj.display %}
   {% if is_own_page %}
{{ obj.short_name }}
{{ "=" * obj.short_name | length }}

``{{ obj.short_name }}({{ obj.args | shorten_type }}){% if obj.return_annotation is not none %} → {{ obj.return_annotation | shorten_type }}{% endif %}``

   {% endif %}
.. py:function:: {{ obj.short_name }}{% if obj.type_params %}[{{ obj.type_params }}]{% endif %}({{ obj.args | tilde_type }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation | tilde_type }}{% endif %}

   {% for (args, return_annotation) in obj.overloads %}
                 {%+ if is_own_page %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}({{ args | tilde_type }}){% if return_annotation is not none %} -> {{ return_annotation | tilde_type }}{% endif %}

   {% endfor %}
   {% for property in obj.properties %}
   :{{ property }}:

   {% endfor %}

   .. container:: import-path

      Import: ``{{ obj.id }}``

   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
