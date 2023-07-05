from django.db import models




class image (models.Model):
    
    image = models.ImageField(upload_to='uploads/', null = True , blank= True)
    classification_results = models.CharField(max_length=100, null=True, blank=True)
    def __str__(self):
        
        return str(self.id)
    