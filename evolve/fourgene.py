import random as r
import matplotlib.pyplot as plt

def triqrt(x):
    if x >= 0:
        return x ** (1/3)
    else:
        return -(-x) ** (1/3)

class Entity:
    def __init__(self, year, parent1, parent2, iffirst=False):
        self.birth = year
        self.isDeath = False
        if not iffirst:
            self.parent_gene1 = parent1.gene
            self.parent_gene2 = parent2.gene
            self.gene = [[0, 0] for _ in range(geneSize)]
            # 매력도 증가 2개, 사망률 증가 2개. 다 우성 / 열성임
            for x in range(geneSize):
                a = parent1.gene[x][r.randint(0, 1)]
                b = parent2.gene[x][r.randint(0, 1)]
                self.gene[x] = [a, b]
        else:
            self.parent_gene1 = None
            self.parent_gene2 = None
            self.gene = [[0, 0] for _ in range(geneSize)]

    @property
    def feature(self):  # 우성 열성을 맞춰서 표현형을 알려줌.
        g = self.gene
        f = ''
        # if self.isDeath:
        #     f += '--'
        for y in range(geneSize//2):
            if sum(g[y * 2]) >= 1:
                f += '1'
            else:
                f += '0'
            if sum(g[y * 2 + 1]) > 1:
                f += '1'
            else:
                f += '0'
        return f

    @property
    def popularity(self):
        f = self.feature
        return  (p[0] ** int(f[0])) * (p[1] ** int(f[1])) * (ConstsLife[3]) * triqrt(-(nowYear - self.birth) + ConstsLife[1])

    @property
    def parents(self):
        pg1 = self.parent_gene1
        pg2 = self.parent_gene2

        rpg1 = ''
        rpg2 = ''
        for x in range(geneSize):
            for y in range(2):
                rpg1 += str(pg1[x][y])
                rpg2 += str(pg2[x][y])
            rpg1 += ' '
            rpg2 += ' '

        return rpg1, rpg2

    @property
    def death(self):
        f = self.feature
        return (p[2] ** int(f[2])) * (p[3] ** int(f[3])) * (ConstsDeath[0] * (triqrt(nowYear-self.birth-ConstsDeath[1]) +ConstsDeath[2]))

def printing(a, l):
    if a == 1:  # Feature
        for x in range(len(l)):
            for y in range(len(l[x])):
                print(l[x][y].feature, end=' ')
            print()

    if a == 2:  # Gene
        for x in range(len(l)):
            for y in range(len(l[x])):
                print(l[x][y].gene, end=' ')
            print()

    if a == 3:
        for x in range(1, len(l)):
            for y in range(len(l[x])):
                print(l[x][y].parents, end=' ')
            print()

    if a == 4:
        for x in range(len(l)):
            print(f'{x}: {len(l[x])} | {deathCount[x]}, {LifeSum[x]} | {DeathSum[x]}')

    if a == 5:
        for x in range(len(l)):
            print(f'{x}: {featureCount[x]}')

    if a == 6:
        for x in range(len(l)):
            print(f'{x}: {geneCount[x]} | {featureCount[x]}, {len(l[x])} | {deathCount[x]}, {LifeSum[x]} | {DeathSum[x]}')

geneSize = 4
# 영향 없음 2개, 매력도 증가 2개, 사망률 증가 2개. 다 우성 / 열성임
p = [1.25, 1.25, 1.25, 1.25]  # 표현형들이 미치는 영향(곱연산)
# 영향 없음 2개, 매력도 증가 2개, 사망률 증가 2개. 다 우성 / 열성임
# ['b', 'r', 'green', 'yellow', 'purple', 'orange']
ConstsLife = [15, 60, 2, 7]  # 최소 나이, 최대 나이, 번식+-, 기초 확률
ConstsDeath = [10, 50, 4]
EndYear = 80
deathCount = [0] * (EndYear + 1)
lifeCount = [0] * (EndYear + 1)
featureCount = [[0] * geneSize for _ in range(EndYear + 1)]
geneCount = [[0] * geneSize for _ in range(EndYear + 1)]

LifeSum = [0] * (EndYear + 1)
DeathSum = [0] * (EndYear + 1)
featureSum = [[0.0] * (EndYear + 1) for _ in range(geneSize)]
geneSum = [[0.0] * (EndYear + 1) for _ in range(geneSize)]

A = Entity(0, None, None, True)
A.gene = [[1, 1], [1, 1], [0, 0], [0, 0]]  # 초기 유전자 결정
B = Entity(0, None, None, True)
B.gene = [[1, 1], [1, 1], [0, 0], [0, 0]]  # 초기 유전자 결정
C = Entity(0, None, None, True)
C.gene = [[0, 0], [0, 0], [1, 1], [1, 1]]  # 초기 유전자 결정
D = Entity(0, None, None, True)
D.gene = [[0, 0], [0, 0], [1, 1], [1, 1]]  # 초기 유전자 결정

entitys = [[A, B, C, D]]
for nowYear in range(1, EndYear + 1):
    # 번식 부분
    print(nowYear)
    entitys.append([])
    for i in range(nowYear):
        if ConstsLife[0] < nowYear - i <= ConstsLife[1]:  # 번식 가능 최대 나이 조건
            group = entitys[i].copy()
            for j in range(ConstsLife[2]):  # +- 몇살까지 번식할지
                group += entitys[i + j]
                if 0 < i - j:
                    group += entitys[i - j]
            for j in range(len(entitys[i])):
                if not entitys[i][j].isDeath:
                    for partner in group:
                        if (not partner.isDeath) and id(partner) != id(entitys[i][j]) and r.randint(1, 100) <= partner.popularity + entitys[i][j].popularity:
                            child = Entity(nowYear, partner, entitys[i][j])
                            entitys[-1].append(child)
                            lifeCount[nowYear] += 1

    for i in range(nowYear):
        for j in range(len(entitys[i])):
            if not entitys[i][j].isDeath:
                z  = r.randint(0, 100)
                if entitys[i][j].death >= z:
                    deathCount[nowYear] += 1
                    entitys[i][j].isDeath = True

    for i in range(len(entitys[nowYear])):
        xyfeature = entitys[nowYear][i].feature
        xygene = entitys[nowYear][i].gene
        for j in range(geneSize):
            featureCount[nowYear][j] += int(xyfeature[j])
            geneCount[nowYear][j] += int(sum(xygene[j]))

    LifeSum[nowYear] = LifeSum[nowYear - 1] + lifeCount[nowYear]
    DeathSum[nowYear] = DeathSum[nowYear - 1] + deathCount[nowYear]
    if lifeCount[nowYear] > 0:
        for i in range(geneSize):
            featureSum[i][nowYear] = featureCount[nowYear][i] / lifeCount[nowYear]


printing(6, entitys)
# 죽음 시스템하고 시각화 만들기

t = range(0, EndYear+1)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
ax1.plot(t, lifeCount, 'b', t, deathCount, 'r')
ax1.set_title("birth and death by year")
ax1.set_xlabel("year")
ax1.set_ylabel("quantity")

ax2.plot(t, LifeSum, 'b', t, DeathSum, 'r')
ax2.set_title("birth and death before year")
ax2.set_xlabel("year")
ax2.set_ylabel("quantity")

pallet = ['b', 'r', 'green', 'yellow']
for i in range(geneSize):
    ax3.plot(t, featureSum[i], pallet[i])
ax3.set_title("gene's evolution")
ax3.set_xlabel("year")
ax3.set_ylabel("quantity")



plt.show()
