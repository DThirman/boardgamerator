import urllib2
import re
with open('games.txt', 'w') as f:
	for i in range(1, 950):
		url = "https://boardgamegeek.com/browse/boardgame/page/" + str(i)
		response = urllib2.urlopen(url)
		page = response.read()

		reg = '<a +href="/boardgame/\d+/.+" +>([^=]*)</a>'

		match = re.findall(reg, page)[50:]
		print i
		
		f.writelines([ x.lower() + '\n' for x in match])