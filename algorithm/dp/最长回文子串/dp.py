# class Solution(object):
#     @staticmethod

def longestPalindrome( s):
    """
    复杂度：2n^2+n
    :type s: str
    :rtype: str
    """
    n = len(s)
    table = [[False]*n for _ in range(n)]
    
    for i in range(n):
        table[i][i] = True
    
    for i in range(n):
        for j in range(n):
            if j == i+1:
                table[i][j] = (s[i]==s[j])
    ans = ""
    for j in range(n):
        for i in range(n):
            if j>i+1:
                table[i][j] = table[i+1][j-1] * (s[i]==s[j])
            if table[i][j] and len(s[i:j+1])>len(ans):
                ans = s[i:j+1]
    
    return ans
        


def main(s):
    return longestPalindrome(s)
    


if __name__ == '__main__':
    s = 'babad'
    ans  = main(s)
    print (ans)
    