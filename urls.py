from django.urls import path

from . import views

urlpatterns = [
    path('alignment/<int:pk>/', views.AlignmentDetailView.as_view(), name='alignment_detail_view'),
]