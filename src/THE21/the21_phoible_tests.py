import pandas as pd
from pandasql import PandaSQL

file_path = '../../datasets/phoible-dev-2.0/data/phoible.csv'
phoible_data = pd.read_csv(file_path)
num_entries = len(phoible_data)
columns = list(phoible_data.columns)
pysql = PandaSQL()

query = '''SELECT GlyphID, Phoneme, SegmentClass, COUNT(*) AS phoneme_count
FROM phoible_data
WHERE SegmentClass != 'tone'
GROUP BY GlyphID
HAVING COUNT(*) > 100
ORDER BY phoneme_count DESC; 
'''

result = pysql(query)
result.to_csv('phonemes_with_count.csv', index=False)

