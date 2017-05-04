from django.conf.urls import include, url
from . import views

urlpatterns = [
	url(r'^search', views.get_name, name='search'),
	url(r'^result', views.index, name='index'),
]