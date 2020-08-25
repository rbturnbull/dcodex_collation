from django.urls import path

from . import views

urlpatterns = [
    path('alignment/<int:pk>/', views.AlignmentDetailView.as_view(), name='alignment_detail_view'),
    path('family/<str:family_siglum>/<str:verse_ref>/', views.alignment_for_family, name='alignment_for_family'),
    path('shift/', views.shift, name='shift'),
    path('shift_to/', views.shift_to, name='shift_to'),
    
]