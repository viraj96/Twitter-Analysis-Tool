from django.db import models

# Create your models here.
class User(models.Model):
	name = models.CharField(max_length = 250)
	def __string__(self):
		return self.name

class Tweets(models.Model):
	tweet = models.ForeignKey(User, on_delete = models.CASCADE)