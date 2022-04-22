/*
Given n numbers and k, whether sum of part of numbers can be k?
Input:
n = 4, {1,5,2,3}
k = 6
Output:
yes 
*/
#include <stdio.h>

int n = 4;
int a[4] = {1,5,10,3};
bool dfs(int sum, int i){
	if (i == n-1) return (a[i]==sum || 0==sum);
	else return (dfs(sum-a[i], i+1) || dfs(sum, i+1));
}


int main(){
	int k;
	scanf("%d",&k);
	printf("Use this numbers to create sum %d:", k);
	for (int i=0; i<n; i++) printf("%d ", a[i]);
	if (dfs(k, 0)) printf("Yes\n");
	else printf("No\n");
}
