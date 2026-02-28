"""
Profile Picture Validator — Django views
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .analyzer import analyze_image


def index(request):
    return render(request, 'validator/index.html')


@csrf_exempt
@require_http_methods(["POST"])
def analyze(request):
    image_file = request.FILES.get('image')
    if not image_file:
        return JsonResponse(
            {'error': 'No image provided. Please POST an image file.'},
            status=400
        )

    if image_file.size > 5 * 1024 * 1024:
        return JsonResponse(
            {'error': 'Image too large. Maximum size is 5 MB.'},
            status=413
        )

    is_valid, score, messages = analyze_image(image_file.read())

    return JsonResponse({
        'valid': is_valid,
        'score': score,
        'messages': messages,
    })
