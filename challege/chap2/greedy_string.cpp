#include <stdio.h>

char s[] = {'A','C','D','B','C','B'};
int n = 6;
// Output head or tail of s until s is empty, 
// to get a minimum string

int main(){
	int a = 0;
	int b = n-1;
	
	while(a <= b){
		bool left = true;
		for (int i=0; a+i <= b; i++){		// if head equals tail, further compare needed
			if (s[a+i] < s[b-i]){
				left = true;
				break;
			}else if (s[a+i] > s[b-i]){
				left = false;
				break;
			}
		}
		if (left) putchar(s[a++]);
		else putchar(s[b--]);	
	}
	putchar('\n');
}		

	
