from django.urls import path
from . import views #import upload_video,display
 
from django.conf.urls.static import static
from django.conf import  settings
from django.urls import path,include
#from majorapp.views import upload_video,display
 
from django.conf.urls.static import static
from django.conf import  settings
 
 
 
urlpatterns = [
    path('',views.home,name='home'),
    path('be',views.be,name='be'),
    path('home',views.home,name='home')
]

 
     

