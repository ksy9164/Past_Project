#include<stdio.h>
#include<iostream>
#include<string.h>
#pragma warning(disable:4996)
#define endtime(x) x%10000
#define starttime(x) x/10000
#include<algorithm>
#include<stdlib.h>
char * subjectname[11] = { "기초, 교양", "문학.언어","역사.철학","정치.경제.사회.세계","과학.기술.자연","예.체능계","인성교육","전공기초교양","학부기초","일반 전공" ,"전공필수"};
int subpoint[11] = { 8,2,2,2,2,2,6,11,23,42,32 };
int msearch(int k, int cnt);
int calculate(void);
int compare(const void *a, const void *b)
{
	const int *da = (const int *)a;
	const int *db = (const int *)b;
 
	return (*da > *db) - (*da < *db);
}
int instant[20];
 
int past[1000000]; // 과거에 수강했던 과목체크
int * answer[30]; //30set의 검색한 과목을 저장함
int * semianswer; //임시 포인터로
int answerpoint[30];
int sbnumber[30];
int minanswer;
int info[1000000]; // 과목코드에 따른 그 과목이 몇번째 배열인지 표시.
 
subject arr[800];	//과목 배열
subject sep[11][150]; //과목별로 분류한 배열
int s1max; // 전공과목 갯수
int s2max; // 교양과목 갯수
int subjectpoint;
int timepoint;
int userinfo[20]; //졸업이수에 의한 현재 수강된 계열현황 표시
 
int userpoint;	// 사용자가 수강한 총학점
 
int mysubject[30]; // 사용자가 수강할 과목 , 코드번호가아닌 arr 번호로 입력받는다
int number; ///mysubject의 갯수
int usertime[7][10]; // 사용자 시간표 시간정보 usertime[0][]... 기준으로 월요일 & usertime[x][0] 여기 배열은 x요일에 몇과목인지 표시 user[0][1] = 10301145이런식으로 표시
 
int mainpoint;
int find;
int proper[15];	///과목계열 선호도
int checkproper[15];///과목 선호도 체크
int empty; // 공강날짜 표시
 
int main()
{
	// 1 들었던 과목 체크
	// 2 현황 표시 (졸업 이수 학점에 의한 현재 학점 여부 표시)
	// 3 선호도 체크 ( 교양에따른 , 공강여부 , 과목에따른 (전공선택 or 전공필수 ) )
	// 4 꼭 듣고싶은 과목 입력
	// 5 전공탐색
	// 6 교양탐색
	int codenumber; // 과거 수강과목
	minanswer = 0;
	int i,j;
	FILE * in = fopen("input.txt", "r");
	FILE * in0 = fopen("input.txt", "r");
	FILE * in1 = fopen("input.txt", "r");
	FILE * in2 = fopen("input.txt", "r"); 
	FILE * in3 = fopen("input.txt", "r");
	FILE * in4 = fopen("input.txt", "r");
	FILE * in5 = fopen("input.txt", "r");
	FILE * in6 = fopen("input.txt", "r");
	FILE * in7 = fopen("input.txt", "r");
	FILE * in8 = fopen("input.txt", "r");
	FILE * in9 = fopen("input.txt", "r");
	FILE * in10 = fopen("input.txt", "r");
	
	fscanf(in, "%d", &s1max);
	/// 인풋이 들어옴 arr로
 
	for (i = 0; i < s1max;i++)
	{
		fscanf(in, "%s", &*arr[i].name);
		fscanf(in, "%s", &*arr[i].propessor);
		fscanf(in, "%d", &arr[i].code);
		info[arr[i].code] = i;
 
		fscanf(in, "%d", &arr[i].date);
		for (j = 0; j < arr[i].date; j++)
		{
			fscanf(in, "%d", &arr[i].day[j]);
			fscanf(in, "%d", &arr[i].time[j]);
		}
		fscanf(in, "%d", &arr[i].major);
		fscanf(in, "%d", &arr[i].id);
		fscanf(in, "%d", &arr[i].limit);
		fscanf(in, "%d", &arr[i].point);
		
	}
	
	
	printf("과거에 수강했던 과목을 입력해주시고, 입력이 끝났으면 0을 입력해주세요\n");
	
	while (1)
	{
		scanf("%d", &codenumber);
		if (codenumber != 0)
		{
			if (past[codenumber] != 1)
			{
				past[codenumber] = 1;
				userinfo[arr[info[codenumber]].id] += arr[info[codenumber]].point;
				userpoint += arr[info[codenumber]].point;
			}
			else
			{
				printf("과목 중복입니다 \n");
			}
		}
		else
			break;
	}
	
	
	/*
	user 에게 현황 보고.
	userinfo 배열에 있는것을 참고해서 현재 수강한 과목에의한 졸업 요건을 보여줌
	
	계열	  이수해야하는학점		이수학점		완성도
0	기초 교양 		9				7				7/9 * 100 ... 
1	문학.언어		2				2
2	역사.철학		2				2
3	정치.경제.사회.세계	2			2
4	과학.기술.자연		2			2
5	예.체능계		2				2
6	인성교육		6				2
7	전공기초교양		17			12
8	학부기초		59				31
9	전공			33				22
	*/
	printf("계열		이수해야하는 학점			이수 학점				완성도 \n");
	for (i = 0; i < 11; i++)
		printf("%s		%d				%d		%.2f\n", subjectname[i],subpoint[i],userinfo[i],(double)userinfo[i]*100 /subpoint[i]);
	
 
	//	과목 점수랑 공강시간 점수 입력
	scanf("%d %d", &subjectpoint, &timepoint);
 
	for (i = 0; i < 11;i++) ///과목선호도 입력
		scanf("%d", &proper[i]);
 
	scanf("%d", &empty); //공강날짜 입력 상관없을시 0을 입력받습니다
 
	// 이제 꼭 수강해야하는 과목을 입력받습니다.
	// 입력후..
	mainpoint = 0;
	int day;
 
	printf("수강을 꼭 해야하는 과목 갯수 입력해주세요\n");
	scanf("%d", &number);
	int tt;
 
	printf("수강을 꼭 해야하는 과목 고유 번호를 입력해주세요\n");
	for (i = 0; i < number; i++)
	{
		scanf("%d", &tt);
		if (past[arr[tt].code] != 1)
		{
			mysubject[i] = tt;
			mainpoint += arr[tt].point;
			for (j = 0; j < arr[tt].date; j++) //주 몇회수업인지
			{
				day = arr[tt].day[j];
				usertime[day][0]++;
				usertime[day][usertime[day][0]] = arr[tt].time[j];
			}
			past[arr[tt].code] = 1;
		}
		else
		{
			i--;
			printf("중복입니다. 다시 입력해주세요 \n");
		}
	}
	printf("몇과목을 찾아드릴까요? \n");
	scanf("%d", &find);
	// mysubject 배열에 number수 만큼 과목이 입력되어있고,(입력시에 과목코드가 아닌 실제 arr의 번호로 입력되어 있다고 가정, past배열에도 표시되어있음)
	// time에 또한 시간이 표시되어있음
 
	
	// 메인과목찾기
 
 
}
