from django.urls import path
from myapp import views


urlpatterns = [
    path("survey", views.surveyView),
    path("surveyprocess", views.surveyProcess),
    path("show", views.surveyShow),
]
