from django.db import models

# Create your models here.

"""
class Jikwon(models.Model):
    jikwonno = models.IntegerField(primary_key=True)
    jikwonname = models.CharField(max_length=10)
    busernum = models.IntegerField()
    jikwonjik = models.CharField(max_length=10, blank=True, null=True)
    jikwonpay = models.IntegerField(blank=True, null=True)
    jikwonibsail = models.DateField(blank=True, null=True)
    jikwongen = models.CharField(max_length=4, blank=True, null=True)
    jikwonrating = models.CharField(max_length=3, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'jikwon'
"""