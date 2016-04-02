import math
import sys

def count_longest_sentence(fname):
    lines = [line.rstrip('\n').split() for line in open(fname)]
    return max([len(line) for line in lines])

print count_longest_sentence('all_questions.txt')
