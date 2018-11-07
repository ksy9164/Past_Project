int calculate(void)
{
	int i,j;
	int tcalpoint;
	int scalpoint;
	tcalpoint = 0;
	scalpoint = 0;
	int a, b, c, d;
	for (i = 0; i < number; i++)
	{
		if (checkproper[arr[mysubject[i]].id] == 0)
		{
			scalpoint += proper[arr[mysubject[i]].id] * arr[mysubject[i]].point;
			checkproper[arr[mysubject[i]].id] = 1;
		}
	}
 
	for (i = 0; i < 7; i++)
	{
		if (usertime[i][0] != 0)
		{
			for (j = 1; j <= usertime[i][0]; j++)
			{
				instant[j - 1] = usertime[i][j];
			}
			qsort(instant, usertime[i][0], sizeof(int), compare);
 
			for (j = 0; j < usertime[i][j] - 1; j++)
			{
				a = (endtime(instant[j]) / 100) * 60 + endtime(instant[j]) % 100;
				b = (starttime(instant[i + 1]) / 100) * 60 + starttime(instant[i + 1]) % 100;
				tcalpoint += b - a;
			}
		}
	}
	tcalpoint = 100 - (tcalpoint / 15);
	for (i = 0; i < 11; i++)
		checkproper[i] = 0;
	
	return (tcalpoint*timepoint) + (scalpoint*subjectpoint);
}
