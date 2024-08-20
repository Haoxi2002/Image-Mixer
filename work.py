file = open('result_long_term_forecast.txt', 'r')
lines = file.readlines()
outfile = open('result_long_term_forecast.csv', 'w')
for i, line in enumerate(lines):
    if i % 3 == 0:
        outfile.write(line.split('_')[1]+',')
    if i % 3 == 1:
        l = line.split(', ')
        outfile.write(l[-3].split(':')[-1]+','+l[-2].split(':')[-1]+','+l[-1].split(':')[-1])
