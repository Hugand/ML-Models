from django.shortcuts import render
from django.http import HttpResponse
from mnist_digit_model.classify_mnist import classify_mnist
from django.core.files.storage import FileSystemStorage
import os
# from .forms import ImageForm

# Create your views here.
def index(request):
    res = classify_mnist("./mnist_digit_model/mnist_6.jpg", "./mnist_digit_model/model/1")
    return HttpResponse("Hello there!<br/>General Kenooobi!!  "+str(res)+"-")

def test(request):
    return render(request, 'core/index.html')

def image_upload(request):
    # return HttpResponse(os.path.abspath("./../mnist_api/templates"))

    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = "." + fs.url(filename)

        mnist_result = classify_mnist(uploaded_file_url, "./mnist_digit_model/model/1")

        os.remove(uploaded_file_url)

        return render(request, 'core/index.html', {
            'uploaded_file_url': uploaded_file_url,
            'mnist_result': mnist_result
        })
    return render(request, 'core/index.html')
