import re
import pandas as pd

file_name = '../submit/predict20201118.csv'
pred = pd.read_csv(file_name)

reg = re.compile(r'(《.*》)')

total_book = pred[pred.Category == 'book'][(pred[pred.Category == 'book'].Privacy.apply(lambda x:re.findall(reg,x)).apply(lambda x:len(x))!=1)]
temp_book = pred[pred.Category == 'book'][(pred[pred.Category == 'book'].Privacy.apply(lambda x:re.findall(reg,x)).apply(lambda x:len(x))!=1)]
bad_data_book = temp_book[temp_book.Privacy.apply(lambda x :('《' in x or '》' in x))]

total_game = pred[pred.Category == 'game'][(pred[pred.Category == 'game'].Privacy.apply(lambda x:re.findall(reg,x)).apply(lambda x:len(x))!=1)]
temp_game = pred[pred.Category == 'game'][(pred[pred.Category == 'game'].Privacy.apply(lambda x:re.findall(reg,x)).apply(lambda x:len(x))!=1)]
bad_data_game = temp_game[temp_game.Privacy.apply(lambda x :('《' in x or '》' in x))]

total_movie = pred[pred.Category == 'movie'][(pred[pred.Category == 'movie'].Privacy.apply(lambda x:re.findall(reg,x)).apply(lambda x:len(x))!=1)]
temp_movie = pred[pred.Category == 'movie'][(pred[pred.Category == 'movie'].Privacy.apply(lambda x:re.findall(reg,x)).apply(lambda x:len(x))!=1)]
bad_data_movie = temp_movie[temp_movie.Privacy.apply(lambda x :('《' in x or '》' in x))]

print(bad_data_book)
print(bad_data_game)
print(bad_data_movie)

def repair_book(bad_data_book,max_len = 20):
    ret = {'ID':[],'Category':[],'Pos_b':[],'Pos_e':[],'Privacy':[]}
    for idx,cat,pos_b,pos_e,privacy in bad_data_book.values:
        if '《' in privacy:
            with open('/Users/mac/Desktop/MLLearn/比赛/CCF隐私/data/test/{}.txt'.format(idx),'r') as f:
                content = f.readline()
            for i in range(1, min(len(content)-pos_e-1,max_len)):
                index  = pos_e + i
                if content[index] in (',.。!！《'):
                    break
                if content[index] == '》':
                    pos_e = index
                    break
            ret['ID'].append(idx)
            ret['Category'].append(cat)
            ret['Pos_b'].append(pos_b)
            ret['Pos_e'].append(pos_e)
            ret['Privacy'].append(content[pos_b:pos_e+1])
        if '》' in privacy:
            with open('/Users/mac/Desktop/MLLearn/比赛/CCF隐私/data/test/{}.txt'.format(idx),'r') as f:
                content = f.readline()
            for i in range(1,min(pos_b,max_len)):
                index  = pos_b -1 - i
                if content[index] in (',.。!！》'):
                    break
                if content[index] == '《':
                    pos_b = index
                    break
            ret['ID'].append(idx)
            ret['Category'].append(cat)
            ret['Pos_b'].append(pos_b)
            ret['Pos_e'].append(pos_e)
            ret['Privacy'].append(content[pos_b:pos_e+1])
    return ret

book = repair_book(bad_data_book)
game = repair_book(bad_data_game)
movie = repair_book(bad_data_movie)

version = '20201122_1'
pred = pred.drop(list(bad_data_book.index))
pred = pred.drop(list(bad_data_game.index))
pred = pred.drop(list(bad_data_movie.index))
book_ = pd.DataFrame(book)
game_ = pd.DataFrame(game)
movie_ = pd.DataFrame(movie)

pred = pred.append(book_,ignore_index=True)
pred = pred.append(game_,ignore_index=True)
pred = pred.append(movie_,ignore_index=True)

pred = pred.sort_values('ID')
pred.to_csv('../submit/predict{}.csv'.format(version),index=None)

