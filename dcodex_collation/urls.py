from django.urls import path

from . import views

urlpatterns = [
    path('alignment/<int:pk>/', views.AlignmentDetailView.as_view(), name='alignment_detail_view'),
    # path('family/<str:family_siglum>/<str:verse_ref>/', views.alignment_for_family, name='alignment_for_family'),
    path('family/<str:family_siglum>/<str:verse_ref>/', views.AlignmentForFamily.as_view(), name='alignment_for_family'),

    path('transition/<str:family_siglum>/<str:verse_ref>/<int:column_rank>/<int:pair_rank>/', views.classify_transition_for_pair, name='classify_transition_for_pair'),
    path('shift/', views.shift, name='shift'),
    path('clear_empty/', views.clear_empty, name='clear_empty'),
    path('shift_to/', views.shift_to, name='shift_to'),
    path('set_transition_type/', views.set_transition_type, name='set_transition_type'),    
    path('set_atext/', views.set_atext, name='set_atext'),    
    path('remove_atext/', views.remove_atext, name='remove_atext'),    
    path('save_atext_notes/', views.save_atext_notes, name='save_atext_notes'),    
    path('alignment-pairwise-comparison/<str:siglum1>/<str:siglum2>/', views.pairwise_comparison, name='alignment-pairwise-comparison'),
    path('alignment-pairwise-comparison/<str:siglum1>/<str:siglum2>/csv', views.disagreement_transitions_csv, name='alignment-pairwise-comparison-csv'),
    path('alignment-pairwise-comparison/', views.ComparisonTableFormView.as_view(), name='alignment-pairwise-comparison-table'),
    path('atext/', views.ATextListView.as_view(), name='atext_list'),    
]