from django import forms

class UserName(forms.Form):
	user_name = forms.CharField(label = 'User Handle ', max_length = 100)