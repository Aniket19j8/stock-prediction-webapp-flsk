s1 = "ccggcctcgggaag"
s2 = "ttgcggacgctagc"
s3 = "tcgggctccccccg"
s4 = "ggggggaaggcgga"
s5 = "tctgtccccccccg"
g = "ggccgcctcccgcgcccctctgtcccctcccgtgttcggcctcgggaagtcggggcggcgggcggcgcgggccgggaggggtcgcctcgggctcaccccgccccagggccgccgggcggaaggcggaggccgagaccagacgcggagccatggccgaggtgttgcggacgctggccg"

#turn G in diccionary
while(i+14<=len(g)):
    dicci[i]=g[i:i+14]
    i+=1

def lcs(X, Y, m, n):
    if m == 0 or n == 0:
        return 0
    elif X[m-1] == Y[n-1]:
        return 1 + lcs(X, Y, m-1, n-1)
    else:
        return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n))

def search_post(str1,dicci):
    for i in range(163):
        if (lcs(str1,dicci[i],len(str1),len(dicci[i])) == 13):
            return (i+1)

print(search_post(s1,dicci))