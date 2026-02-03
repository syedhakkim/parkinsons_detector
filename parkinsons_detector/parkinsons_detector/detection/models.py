from django.db import models

class Drawing(models.Model):
    image = models.ImageField(upload_to='drawings/')
    prediction = models.CharField(max_length=20, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Drawing {self.id} - {self.prediction}"