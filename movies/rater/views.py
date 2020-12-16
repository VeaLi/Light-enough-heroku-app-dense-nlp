from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .apps import RaterConfig


class call_model(APIView):
    def get(self, request):
        if request.method == 'GET':

            text = request.GET.get('text')
            response = RaterConfig.model.predict([text])
            response = {
                'impression': str(response[0]),
                'rating': str(response[1])
            }
            print(response)

            return JsonResponse(response)
