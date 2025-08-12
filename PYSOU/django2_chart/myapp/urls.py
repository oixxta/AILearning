from django.urls import path
from myapp import views


urlpatterns = [
    path('', views.show, name='show'),
]
