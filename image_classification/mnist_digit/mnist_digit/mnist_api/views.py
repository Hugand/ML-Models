from django.shortcuts import render
from django.http import HttpResponse
from mnist_digit_model.classify_mnist import classify_mnist
from django.core.files.storage import FileSystemStorage
import os
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
def index(request):
    return render(request, 'core/index.html')

def test(request):
    return render(request, 'core/index.html')

@csrf_exempt
def image_upload(request):
    if request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = "." + fs.url(filename)

        mnist_result = classify_mnist(uploaded_file_url, "./mnist_digit_model/model/1")
        os.remove(uploaded_file_url)

        return HttpResponse(mnist_result)
    return HttpResponse("error")
