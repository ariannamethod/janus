#!/usr/bin/env python3
"""
NanoJanus — Bi-directional associative reasoning in the terminal.

Architecture (matching nanojanus.html / penelope.c from 1984 repo):
  - Real BPE input tokenizer (2048 vocab, 1792 merges from penelope.c)
  - Word-level output (NWORDS from nanojanus.txt)
  - Dual embed: embed_in_A/B[BPE_VOCAB * DIM] + embed_out_A/B[NWORDS * DIM]
  - No weight tying between input and output embeddings
  - RRPRAM per-step: pool_context -> Wr -> RMSNorm -> SwiGLU -> logits(embed_out)
  - Dario equation overlay on top of learned logits
  - Dual weight matrices blended by calendar drift
  - 12 bi-directional steps (simultaneous forward + backward)
  - Kuramoto chambers (6 coupled oscillators)
  - MetaJanus birth date mathematical "self"
  - Chuck optimizer for training
"""

import argparse
import math
import os
import pickle
import random
import re
import sys
import time

# ===================================================================
# CONSTANTS
# ===================================================================
STEPS = 12
DIM = 64
HDIM = 128
BPE_VOCAB = 2048
BPE_MERGES = 1792

# ===================================================================
# BPE MERGE TABLE (from penelope.c / 1984 repo -- 1792 merges)
# ===================================================================
BPE_TABLE = [
(115,32),(101,32),(46,32),(105,110),(105,256),(101,114),(111,110),(116,104),(116,32),(101,110),(97,110),(116,105),(104,260),(101,115),(121,32),(258,268),(100,32),(111,114),(259,103),(97,114),
(97,108),(274,32),(267,262),(111,117),(101,256),(114,101),(111,32),(105,116),(97,116),(58,32),(111,109),(115,116),(100,105),(101,108),(104,97),(114,269),(112,261),(261,32),(263,257),(266,272),
(278,32),(111,119),(97,99),(105,115),(266,32),(44,32),(39,256),(276,32),(108,105),(265,99),(114,97),(116,282),(117,114),(101,272),(105,109),(102,102),(101,120),(101,99),(101,109),(102,32),
(102,273),(114,111),(101,116),(10,10),(97,285),(113,285),(10,320),(46,319),(323,321),(117,110),(97,32),(117,108),(101,118),(265,264),(290,264),(119,330),(105,99),(265,116),(275,257),(284,116),
(97,115),(103,104),(97,296),(63,322),(288,311),(105,32),(115,105),(99,262),(110,111),(112,291),(305,257),(101,271),(100,101),(111,108),(105,108),(286,101),(283,270),(263,101),(97,98),(101,100),
(115,351),(97,103),(99,286),(275,105),(115,117),(262,32),(340,261),(114,117),(269,115),(111,315),(97,112),(119,104),(262,257),(105,114),(108,270),(98,101),(115,99),(109,101),(98,257),(265,32),
(259,32),(115,289),(267,118),(114,105),(111,99),(115,104),(267,109),(97,109),(112,317),(344,264),(113,117),(105,263),(263,300),(335,261),(109,273),(266,110),(119,273),(112,111),(101,264),(279,108),
(121,258),(395,272),(97,278),(267,99),(353,270),(119,101),(359,391),(402,326),(45,32),(109,32),(273,32),(266,99),(97,100),(324,331),(257,260),(121,279),(121,271),(111,263),(263,277),(119,387),
(112,108),(276,352),(290,112),(102,101),(101,258),(105,100),(100,97),(279,264),(117,109),(100,117),(104,32),(337,264),(292,105),(115,271),(116,114),(100,256),(100,277),(99,104),(109,270),(107,32),
(276,108),(100,111),(116,256),(109,389),(103,117),(118,261),(115,112),(105,264),(99,111),(108,257),(294,115),(258,403),(119,397),(422,270),(411,32),(98,117),(258,341),(99,300),(121,302),(112,275),
(116,111),(114,297),(116,306),(269,256),(110,282),(275,32),(105,427),(258,294),(328,261),(98,318),(316,32),(102,259),(115,262),(114,286),(97,264),(277,295),(115,107),(99,108),(304,102),(348,112),
(117,115),(265,115),(110,364),(100,282),(101,301),(312,428),(108,32),(356,414),(292,468),(116,117),(281,99),(415,423),(99,275),(116,108),(112,32),(342,361),(105,287),(103,103),(111,111),(308,257),
(327,116),(259,116),(265,267),(261,257),(297,110),(263,293),(297,32),(335,265),(115,258),(112,273),(97,107),(98,317),(390,257),(263,261),(97,117),(430,270),(377,102),(103,110),(377,315),(451,264),
(109,111),(362,329),(279,115),(355,271),(275,272),(118,472),(425,507),(522,521),(109,462),(287,363),(101,103),(306,501),(527,388),(278,303),(269,271),(278,256),(524,374),(112,304),(116,297),(534,520),
(356,382),(102,105),(263,259),(536,280),(511,307),(273,105),(367,499),(292,418),(325,100),(110,100),(477,338),(543,256),(39,264),(298,426),(437,280),(263,269),(401,375),(446,546),(465,552),(314,111),
(99,101),(390,110),(508,388),(100,269),(362,346),(274,271),(98,413),(439,256),(551,257),(263,470),(400,334),(121,394),(339,294),(448,540),(474,257),(458,424),(381,105),(266,100),(314,32),(115,380),
(575,105),(562,32),(261,103),(484,417),(339,523),(118,105),(104,349),(266,103),(409,260),(357,257),(386,269),(112,114),(358,316),(105,122),(104,502),(111,112),(111,441),(99,296),(555,529),(383,257),
(101,119),(116,354),(262,103),(557,277),(98,105),(116,261),(281,408),(102,327),(99,366),(101,549),(316,109),(277,115),(475,265),(101,302),(419,289),(99,114),(512,45),(345,329),(119,97),(102,469),
(580,454),(263,260),(270,260),(371,277),(615,32),(116,259),(259,264),(279,114),(109,266),(290,256),(284,257),(378,257),(115,257),(98,108),(116,289),(287,114),(291,112),(111,100),(284,309),(261,118),
(103,259),(34,32),(101,275),(349,117),(115,595),(312,116),(103,306),(407,257),(479,450),(112,97),(104,289),(632,262),(109,329),(110,297),(265,578),(516,378),(550,385),(382,257),(109,262),(467,431),
(392,435),(282,260),(102,306),(115,121),(324,590),(456,282),(283,256),(259,459),(328,265),(312,492),(342,262),(102,298),(398,256),(447,655),(263,574),(345,333),(611,299),(99,281),(107,257),(104,293),
(266,264),(292,102),(505,116),(343,102),(288,115),(369,32),(283,514),(481,305),(333,271),(457,256),(313,116),(584,294),(108,266),(292,606),(260,385),(660,644),(121,263),(105,513),(115,308),(688,440),
(538,107),(677,313),(112,104),(293,263),(340,332),(279,337),(373,394),(440,350),(488,114),(99,334),(115,418),(415,32),(349,111),(280,307),(116,265),(116,370),(260,104),(332,303),(287,259),(304,674),
(500,32),(110,313),(646,112),(97,259),(99,97),(481,346),(373,288),(327,461),(120,105),(299,301),(119,259),(537,289),(581,596),(99,379),(353,681),(361,331),(108,598),(706,280),(266,724),(650,270),
(281,108),(278,258),(556,112),(104,310),(280,334),(651,338),(360,99),(115,101),(287,284),(476,264),(734,318),(630,482),(111,311),(328,293),(392,107),(99,266),(358,416),(102,97),(299,405),(436,256),
(413,293),(623,99),(586,531),(105,315),(308,277),(291,262),(263,32),(345,346),(485,281),(452,569),(708,103),(372,105),(610,32),(571,97),(279,545),(298,278),(455,399),(116,271),(559,729),(116,641),
(525,99),(381,397),(283,117),(103,266),(98,336),(107,649),(109,259),(100,760),(273,779),(309,376),(109,314),(589,280),(631,284),(265,117),(333,370),(727,272),(489,396),(118,257),(288,486),(280,102),
(108,101),(772,723),(274,301),(115,313),(291,757),(328,375),(356,368),(119,283),(425,99),(639,278),(774,374),(104,111),(266,101),(717,364),(366,533),(588,597),(115,264),(419,461),(775,495),(809,275),
(109,275),(310,496),(817,808),(104,257),(274,258),(695,585),(310,678),(510,263),(662,716),(664,277),(358,112),(343,767),(376,283),(818,518),(324,806),(803,478),(582,432),(259,284),(325,811),(98,770),
(732,293),(525,493),(98,273),(460,836),(109,308),(280,436),(333,338),(509,410),(544,293),(822,676),(837,108),(100,500),(272,365),(355,258),(362,790),(371,636),(463,791),(766,713),(834,445),(274,322),
(498,116),(97,256),(642,442),(105,102),(288,714),(710,491),(635,332),(778,338),(99,369),(784,787),(99,755),(102,363),(298,485),(393,287),(420,460),(604,764),(694,667),(700,496),(744,480),(258,539),
(269,438),(101,107),(331,690),(363,621),(372,879),(39,32),(267,337),(277,661),(301,300),(309,620),(541,842),(814,404),(860,593),(886,535),(45,570),(284,280),(295,815),(380,634),(602,663),(625,797),
(792,843),(878,567),(107,259),(406,839),(443,577),(483,487),(528,771),(535,894),(553,365),(553,895),(613,899),(617,874),(682,850),(715,832),(761,407),(783,907),(800,841),(828,884),(830,904),(835,359),
(854,892),(858,883),(861,913),(865,908),(882,896),(887,909),(889,897),(893,903),(900,916),(901,917),(905,921),(906,671),(911,912),(918,922),(919,928),(920,929),(923,902),(925,931),(926,933),(930,932),
(934,927),(392,431),(109,97),(393,622),(115,805),(263,258),(370,404),(384,118),(489,121),(691,721),(852,935),(360,493),(386,417),(102,336),(560,554),(851,110),(99,308),(898,848),(936,946),(367,657),
(424,300),(687,950),(704,270),(924,121),(107,270),(409,448),(583,108),(867,788),(103,685),(99,833),(114,104),(269,669),(324,453),(406,547),(961,450),(295,547),(307,483),(439,301),(463,560),(292,624),
(517,962),(608,432),(840,960),(949,965),(396,368),(480,756),(563,558),(564,758),(607,829),(728,885),(844,880),(846,709),(942,400),(974,563),(977,731),(979,471),(115,325),(116,363),(310,697),(368,810),
(373,656),(414,985),(532,475),(532,872),(565,821),(566,640),(652,973),(654,449),(658,996),(845,1000),(871,989),(888,964),(939,972),(963,984),(967,983),(969,1001),(971,1002),(976,1010),(978,986),(980,999),
(981,998),(987,1006),(988,1008),(990,1004),(991,1009),(995,269),(997,1013),(1005,1017),(1007,1014),(1011,1022),(1012,1019),(1015,1016),(1018,1023),(1020,1028),(1024,1027),(1025,1029),(1026,1021),(1031,1032),(400,777),(736,398),
(824,953),(970,747),(504,539),(702,670),(748,699),(855,954),(873,618),(966,692),(336,576),(446,863),(464,478),(466,705),(473,1046),(528,542),(542,566),(558,1048),(619,831),(725,994),(763,982),(785,1042),
(802,955),(866,1047),(940,1038),(1030,941),(1034,371),(1036,718),(1037,1056),(1039,1050),(1040,1053),(1045,1057),(1052,1055),(1054,1058),(1063,1049),(1065,1051),(1066,1061),(1067,1070),(343,776),(672,260),(1035,572),(1059,1033),
(310,533),(753,350),(339,1069),(947,876),(875,1071),(600,853),(659,545),(544,261),(1043,405),(1060,1080),(1064,944),(102,494),(568,1075),(827,518),(421,856),(794,711),(503,737),(742,99),(294,937),(428,633),
(1044,668),(110,101),(307,605),(712,956),(280,801),(288,300),(291,426),(401,877),(653,365),(720,1101),(864,1105),(432,847),(449,1099),(453,768),(726,1107),(1072,264),(1091,683),(1104,1108),(1113,1111),(106,745),
(115,267),(258,599),(281,383),(404,517),(487,730),(564,910),(567,1094),(675,735),(733,1123),(780,299),(795,1102),(798,825),(870,1106),(948,1098),(951,1127),(958,1096),(1076,1126),(1079,1110),(1081,1125),(1084,1124),
(1095,1117),(1100,1120),(1109,1119),(1112,1128),(1121,1137),(1122,1131),(1129,1136),(1130,1133),(1132,1143),(1135,406),(1138,1142),(1139,1145),(1140,1134),(1146,1144),(281,276),(992,449),(105,262),(339,1114),(1147,1092),(1154,1141),
(346,260),(637,268),(121,256),(265,399),(759,434),(99,273),(509,366),(576,303),(112,101),(97,627),(679,421),(121,115),(345,491),(751,548),(275,270),(868,303),(119,275),(278,271),(384,804),(823,1159),
(592,696),(103,789),(108,97),(698,368),(1177,259),(337,116),(498,303),(579,260),(276,103),(647,628),(503,296),(112,336),(479,385),(746,270),(108,111),(115,97),(110,459),(769,302),(409,1160),(281,386),
(968,434),(103,111),(358,109),(108,259),(354,423),(447,569),(1116,108),(538,435),(571,326),(283,303),(701,32),(1087,116),(273,270),(261,271),(952,114),(341,1188),(494,272),(1207,587),(256,334),(109,333),
(299,109),(665,1182),(813,365),(119,266),(112,389),(276,271),(1220,110),(299,264),(285,34),(116,302),(279,110),(357,103),(341,1203),(378,352),(281,118),(289,270),(1068,1228),(332,32),(1153,1211),(325,99),
(341,1149),(109,506),(588,264),(269,258),(1232,1085),(304,103),(1074,490),(1082,469),(98,313),(1155,1236),(316,464),(799,308),(693,273),(103,114),(572,102),(360,98),(273,100),(281,417),(283,454),(269,116),
(283,412),(1210,329),(98,114),(98,270),(526,282),(360,112),(116,293),(419,275),(101,112),(117,287),(110,548),(121,277),(261,116),(112,117),(116,379),(265,272),(354,108),(467,272),(1093,364),(259,1247),
(288,103),(1276,1205),(116,506),(121,262),(433,275),(103,318),(276,370),(114,1206),(305,101),(312,112),(398,271),(46,1157),(101,336),(317,108),(118,276),(299,360),(104,101),(116,309),(261,256),(433,350),
(442,369),(826,318),(1219,438),(100,265),(104,609),(261,258),(279,427),(289,108),(452,1273),(474,410),(108,412),(263,1283),(269,264),(277,109),(457,32),(614,256),(327,264),(265,100),(265,103),(325,105),
(310,943),(313,104),(453,374),(102,286),(816,107),(109,117),(556,355),(110,749),(345,666),(277,260),(310,869),(348,408),(1304,959),(110,526),(286,32),(345,115),(510,456),(703,264),(1161,486),(1299,105),
(411,114),(673,891),(343,116),(383,600),(283,396),(298,738),(401,275),(98,1227),(115,862),(304,99),(1195,369),(452,838),(890,1073),(1078,765),(1295,100),(1310,1148),(1347,1351),(433,583),(444,112),(765,1086),
(1041,1328),(367,375),(371,1279),(497,261),(1358,272),(505,264),(100,279),(287,266),(1362,98),(269,881),(314,329),(1103,1271),(1180,1231),(531,334),(393,115),(1217,100),(1317,266),(1234,1245),(354,573),(488,98),
(408,288),(1336,32),(100,313),(464,121),(1174,1229),(1375,361),(116,258),(308,110),(1381,1213),(739,32),(380,107),(384,102),(1327,1199),(1374,262),(348,1168),(1274,603),(354,445),(645,267),(1176,277),(1350,104),
(115,301),(594,1343),(915,740),(104,573),(295,434),(344,399),(101,311),(782,100),(298,104),(1115,434),(1243,257),(1372,1216),(287,1118),(97,118),(110,293),(430,1267),(648,1291),(292,738),(1395,1212),(117,281),
(1399,445),(105,304),(273,387),(343,621),(541,636),(1184,1418),(1341,116),(1419,117),(101,490),(102,332),(342,513),(97,272),(103,101),(276,754),(1179,1376),(100,271),(259,1410),(1332,45),(112,281),(1433,1334),
(316,749),(492,506),(1202,482),(106,111),(455,116),(741,260),(1238,122),(299,104),(689,643),(1354,1309),(114,257),(283,271),(1090,270),(1187,264),(32,260),(106,286),(109,1437),(416,266),(1089,1192),(307,374),
(1198,283),(1290,117),(1339,296),(288,118),(304,287),(1285,686),(1421,405),(119,350),(259,338),(444,513),(1235,1268),(381,297),(1261,1361),(299,1152),(1166,1156),(384,99),(786,1208),(1089,478),(1346,280),(1445,1407),
(298,257),(437,269),(1289,108),(384,629),(607,862),(110,502),(115,796),(401,369),(407,347),(1408,1480),(119,114),(1296,303),(372,379),(373,266),(407,410),(418,112),(782,310),(100,257),(633,270),(39,1446),
(276,614),(444,1252),(647,115),(673,1165),(98,276),(115,881),(497,289),(752,318),(112,105),(289,105),(399,32),(752,312),(781,256),(1502,1241),(267,115),(350,352),(510,116),(1281,256),(332,426),(357,410),
(612,1364),(111,115),(379,118),(441,115),(263,314),(1272,347),(384,116),(1222,256),(100,1118),(280,295),(348,102),(1240,1355),(109,421),(258,460),(312,313),(1320,318),(1530,117),(111,267),(1335,303),(1342,277),
(100,314),(119,262),(739,271),(1251,1488),(1321,372),(1442,368),(1496,1158),(121,301),(1003,599),(110,261),(372,1478),(594,1509),(975,684),(1540,445),(98,281),(1394,1487),(279,256),(316,405),(456,1428),(669,959),
(672,299),(1163,722),(1329,1533),(100,1167),(1559,102),(108,494),(115,111),(266,116),(281,106),(367,1514),(102,108),(102,350),(110,318),(393,497),(111,332),(348,97),(1041,1555),(100,318),(372,104),(1078,1201),
(1344,257),(116,276),(278,302),(608,100),(1201,1086),(1477,1266),(110,262),(1549,1472),(99,336),(281,284),(283,302),(357,281),(437,1330),(680,366),(1275,352),(1463,482),(99,107),(109,257),(465,1262),(416,1288),
(586,296),(1263,302),(1482,1424),(101,263),(108,396),(109,332),(115,635),(260,259),(269,666),(99,349),(103,366),(276,693),(1430,593),(98,389),(111,98),(263,1302),(298,99),(1257,1497),(1314,357),(1588,1546),
(270,295),(316,99),(1492,1429),(291,639),(1589,1569),(447,838),(685,1148),(1554,509),(1621,1622),(115,463),(298,101),(975,329),(1539,112),(327,275),(103,457),(110,462),(116,1308),(313,296),(750,890),(1244,286),
(1380,1333),(1422,643),(1459,273),(1557,326),(366,112),(703,1225),(1197,276),(269,301),(816,379),(1162,1223),(327,438),(360,311),(281,1427),(290,793),(353,121),(355,327),(1571,762),(1574,1651),(102,379),(263,271),
(443,260),(1466,719),(1634,1500),(108,526),(287,1386),(291,308),(582,405),(1660,1662),(1661,486),(386,275),(1606,32),(259,118),(298,378),(393,342),(396,294),(469,299),(1373,1352),(1638,99),(311,101),(342,1493),
(696,256),(807,264),(1650,1495),(109,121),(273,271),(1378,1469),(314,112),(1249,279),(1598,1653),(287,1184),(298,264),(344,1685),(1467,699),(1677,1278),(98,1383),(263,375),(305,347),(938,376),(1090,454),(1613,1464),
(1687,105),(473,336),(786,541),(1187,115),(1237,280),(110,596),(291,1646),(301,294),(310,722),(392,272),(1440,1545),(1483,272),(1665,601),(98,457),(109,299),(117,100),(333,256),(378,347),(451,1592),(1709,115),
(312,290),(409,550),(490,260),(781,277),(1250,116),(258,605),(310,1168),(372,359),(438,307),(1331,495),(98,379),(354,115),(612,705),(1701,32),(104,1265),(115,394),(807,287),(1151,1723),(1387,1604),(119,859),
(259,1297),(261,105),(298,107),(313,438),(367,289),(442,1303),(592,1740),(1083,287),(1379,368),(1735,341),(102,369),(372,281),(608,431),(1246,271),(1284,1088),(1370,342),(259,307),(488,630),(597,256),(1435,264),
(1449,1452),(281,103),(1179,1609),(1465,105),(1714,394),(373,300),(41,271),(281,109),(298,435),(355,103),(1326,406),(1479,574),(99,121),(260,577),(336,262),(1461,668),(1657,258),(1696,326),(1715,293),(350,108),
(579,1632),(700,1312),(1202,108),(1528,1348),(312,1322),(325,593),(381,283),(413,1301),(570,444),(601,271),(1250,264),(1269,381),(1532,627),(1776,1702),(261,110),(283,121),(308,410),(336,1504),(436,32),(476,257),
(1384,622),(1793,306),(111,302),(116,495),(118,380),(358,1393),(433,313),(591,259),(680,440),(1800,354),(103,336),(105,112),(276,1167),(472,1264),(799,262),(1666,554),(1780,256),(1809,399),(115,109),(263,701),
(624,859),(1420,417),(1524,256),(1607,648),(1745,1699),(1788,1560),(1805,1629),(98,306),(118,293),(515,276),(689,1165),(1083,437),(1097,355),(1690,423),(102,117),(118,349),(382,626),(1175,1340),(1582,45),(121,638),
(288,372),(439,115),(781,108),(993,812),(1178,119),(1293,1577),(1516,264),(1664,296),(116,277),(118,289),(295,279),(328,421),(331,368),(442,1626),(687,1242),(703,116),(1176,471),(1803,1152),(1850,554),(281,1164),
(393,259),(443,1761),(617,515),(915,280),(1093,459),(1371,1648),(1468,1683),(1717,1857),(1804,299),(259,99),(259,1801),(266,310),(298,296),(342,287),(1162,270),(1186,442),(1226,272),(1240,1580),(1260,1652),(1520,271),
(1725,307),(1777,404),(1806,304),(97,1172),(98,1505),(105,118),(325,116),(629,347),(1541,260),(1789,334),(1802,107),(98,261),(99,383),(258,1300),(280,1360),(344,263),(436,412),(523,270),(682,260),(1640,638),(1742,405),
(357,719),(381,275),(453,1896),(464,692),(568,1596),(702,1771),(1315,1519),(1411,1837),(1432,1846),(1819,554),(1838,1765),(1848,1368),(1856,1724),(1863,1455),(1876,1902),(1911,1899),(99,279),(267,373),(331,1318),(331,1368),
(358,280),(466,1858),(601,1529),(659,287),(725,1565),(1171,1457),(1360,1916),(1365,1753),(1397,585),(1444,1923),(1451,282),(1474,1719),(1485,1924),(1656,1877),(1668,754),(1703,1904),(1704,626),(1728,1151),(1772,1778),(1820,1705),
(1826,1931),(1833,619),(1894,1935),(1905,1919),(1906,1940),(1908,1921),(1909,1941),(1912,1938),(1918,1930),(1926,299),(1928,1942),(1939,1932),(1943,1946),(1945,1944),(1947,1948),(306,103),(455,711),(587,310),(1230,101),(1242,1603),
(1810,100),(1832,295),(1956,1958),(265,256),(314,684),(473,117),(695,357),(731,1318),(733,294),(1326,293),(1456,1412),(1507,1721),(1562,301),(1601,457),(1658,1490),(1748,1953),(1767,295),(1811,670),(1972,1964),(45,1237),
(99,1265),(107,101),(612,1413),(1257,1822),(1292,276),(1371,602),(1475,256),(1537,548),(1670,1974),(1681,1976),(1769,1973),(1787,1890),(1825,1969),(1959,1968),(1965,1783),(1980,1985),(1987,1499),(1988,1992),(1990,1991),(1994,1993),
(309,1259),(343,99),(380,114),(408,1312),(1486,283),(1512,286),(1747,375),(1901,1949),(1995,1915),(455,1808),(1097,309),(1258,295),(1388,609),(1498,420),(1879,265),(1996,1849),(383,32),(568,2005),(638,110),(1185,307),
(1708,1348),(101,603),(105,348),(109,684),(116,119),(121,45),(317,441),(1277,1618),(1367,1453),(1619,1369),(1784,549),(1841,435),(1954,1170),(98,1494),(455,267),(587,298),(1402,686),(97,1506),(498,281),(1630,762),
(1716,476),(1982,302),(103,394),(104,638),(108,354),(276,105),(304,109),(312,1324),(613,1986),(742,1322),(1074,112),
]

assert len(BPE_TABLE) == BPE_MERGES, \
    f'BPE_TABLE has {len(BPE_TABLE)} entries, expected {BPE_MERGES}'

# ===================================================================
# BPE TOKENIZER (real byte-pair encoding, from penelope.c)
# ===================================================================

def bpe_encode(text):
    seq = [c for c in text.lower().encode('ascii', errors='ignore')]
    for m in range(BPE_MERGES):
        left, right = BPE_TABLE[m]
        new_id = 256 + m
        out = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == left and seq[i+1] == right:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return seq

# ===================================================================
# VOCABULARY
# ===================================================================
SUFFIXES = [
    'ing', 'tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ful',
    'less', 'ous', 'ive', 'ity', 'ary', 'ery', 'ory', 'al',
    'ly', 'er', 'ed', 'es', 'en', 's',
]

WORDS = []
VOCAB_LENS = []
NWORDS = 0
vocab_bpe = []  # precomputed BPE encoding for each vocabulary word


def load_vocabulary(vocab_path):
    """Load vocabulary from nanojanus.txt (one word per line)."""
    global WORDS, VOCAB_LENS, NWORDS, vocab_bpe
    with open(vocab_path, 'r') as f:
        WORDS = [line.strip() for line in f if line.strip()]
    VOCAB_LENS = [len(w) for w in WORDS]
    NWORDS = len(WORDS)
    # Precompute BPE encoding for each word in vocabulary
    vocab_bpe = [bpe_encode(w) for w in WORDS]


def try_stem(word):
    """Try stripping known suffixes to find a vocab match."""
    for suf in SUFFIXES:
        if len(word) > len(suf) + 2 and word.endswith(suf):
            stem = word[:-len(suf)]
            if stem in WORDS:
                return WORDS.index(stem)
    return -1


def tokenize_words(text):
    """Word-level tokenizer for finding context/origin words."""
    words = re.sub(r'[^a-z\s]', '', text.lower()).split()
    ids = []
    for w in words:
        if not w:
            continue
        if w in WORDS:
            ids.append(WORDS.index(w))
            continue
        stem_idx = try_stem(w)
        if stem_idx >= 0:
            ids.append(stem_idx)
    return ids


# ===================================================================
# CALENDAR DRIFT (Metonic cycle, Gregorian vs Hebrew)
# ===================================================================
AM_ANNUAL_DRIFT = 11.25
AM_GREGORIAN_YEAR = 365.25
AM_METONIC_YEARS = 19
AM_METONIC_LEAPS = 7
AM_MAX_UNCORRECTED = 33.0
METONIC_LEAP_YEARS = [3, 6, 8, 11, 14, 17, 19]

# Epoch: 1 Tishrei 5785 = October 3, 2024, 12:00:00 UTC
EPOCH_TIMESTAMP = 1727956800.0


def calendar_days_since_epoch():
    return int((time.time() - EPOCH_TIMESTAMP) / 86400)


def calendar_cumulative_drift(days):
    years = days / AM_GREGORIAN_YEAR
    base_drift = years * AM_ANNUAL_DRIFT
    full_cycles = int(years / AM_METONIC_YEARS)
    corrections = full_cycles * AM_METONIC_LEAPS * 30
    partial = years % AM_METONIC_YEARS
    yic = int(partial) + 1
    for i in range(AM_METONIC_LEAPS):
        if METONIC_LEAP_YEARS[i] <= yic:
            corrections += 30
    return base_drift - corrections


def calendar_dissonance(days):
    drift = calendar_cumulative_drift(days)
    raw = abs(drift % AM_MAX_UNCORRECTED) / AM_MAX_UNCORRECTED
    return max(0.0, min(1.0, raw))


# ===================================================================
# METAJANUS -- birth date snapshot
# ===================================================================
class MetaJanus:
    def __init__(self):
        d = calendar_days_since_epoch()
        self.birth_days = d
        self.birth_drift = calendar_cumulative_drift(d)
        self.birth_dissonance = calendar_dissonance(d)
        self.birth_time = time.time()
        self.prophecy_accuracy = 0.5
        self.total_predictions = 0


META = None  # initialized after vocabulary loads


def personal_dissonance():
    now_drift = calendar_cumulative_drift(calendar_days_since_epoch())
    return max(0.0, min(1.0, abs(now_drift - META.birth_drift) / AM_MAX_UNCORRECTED))


# ===================================================================
# AML PHYSICS
# ===================================================================
prophecy_debt = 0.0
destiny_bias = 0.1
wormhole = 0.02
resonance_field = 0.5
trauma = 0.0


def compute_prophecy_debt(scores, chosen_idx):
    if not scores:
        return 0.0
    mx = max(s['score'] for s in scores)
    chosen = scores[chosen_idx]['score'] if chosen_idx < len(scores) else 0.0
    diff = mx - chosen
    return diff / (diff + 1) if diff > 0 else 0.0


# ===================================================================
# KURAMOTO CHAMBERS (6 coupled oscillators)
# ===================================================================
CH_FEAR = 0
CH_LOVE = 1
CH_RAGE = 2
CH_VOID = 3
CH_FLOW = 4
CH_COMPLEX = 5
CH_N = 6

chambers = [0.0] * CH_N
CH_DECAY = [0.95, 0.95, 0.93, 0.96, 0.94, 0.97]


def update_chambers(step_idx):
    global chambers
    depth = step_idx / STEPS
    if depth < 0.33:
        chambers[CH_FLOW] += 0.05
    elif depth < 0.66:
        chambers[CH_FEAR] += 0.04
    else:
        chambers[CH_VOID] += 0.05
    if depth > 0.75:
        chambers[CH_COMPLEX] += 0.03
    if trauma > 0.3:
        chambers[CH_RAGE] += 0.04
    K = 0.02
    old = list(chambers)
    for i in range(CH_N):
        for j in range(CH_N):
            if i != j:
                chambers[i] += K * math.sin(old[j] - old[i])
        chambers[i] = max(0.0, min(1.0, chambers[i] * CH_DECAY[i]))


# ===================================================================
# DUAL WEIGHT MATRICES -- BPE input + word output (no weight tying)
#   embed_in_A/B:  [BPE_VOCAB * DIM]  -- BPE subword input embeddings
#   embed_out_A/B: [NWORDS * DIM]     -- word output embeddings
#   Per-step RRPRAM: Wr, RMSNorm, SwiGLU gate/up/down
# ===================================================================
embed_in_A = []
embed_in_B = []
embed_out_A = []
embed_out_B = []
step_wr_a = []
step_wr_b = []
step_rms_a = []
step_rms_b = []
step_gate_a = []
step_up_a = []
step_down_a = []
step_gate_b = []
step_up_b = []
step_down_b = []


def _rand_vec(size, scale):
    return [(random.random() - 0.5) * scale for _ in range(size)]


def _ones_vec(size):
    return [1.0] * size


def init_weights():
    """Initialize all weight matrices (dual BPE input + word output + RRPRAM)."""
    global embed_in_A, embed_in_B, embed_out_A, embed_out_B
    global step_wr_a, step_wr_b, step_rms_a, step_rms_b
    global step_gate_a, step_up_a, step_down_a
    global step_gate_b, step_up_b, step_down_b

    scale = math.sqrt(2.0 / DIM) * 0.02
    scale_bpe = math.sqrt(2.0 / BPE_VOCAB) * 0.02

    # Input embeddings: BPE_VOCAB * DIM
    embed_in_A = _rand_vec(BPE_VOCAB * DIM, scale_bpe)
    embed_in_B = _rand_vec(BPE_VOCAB * DIM, scale_bpe)

    # Output embeddings: NWORDS * DIM (no weight tying)
    embed_out_A = _rand_vec(NWORDS * DIM, scale)
    embed_out_B = _rand_vec(NWORDS * DIM, scale)

    step_wr_a = []
    step_wr_b = []
    step_rms_a = []
    step_rms_b = []
    step_gate_a = []
    step_up_a = []
    step_down_a = []
    step_gate_b = []
    step_up_b = []
    step_down_b = []

    for _ in range(STEPS):
        step_wr_a.append(_rand_vec(DIM * DIM, scale))
        step_wr_b.append(_rand_vec(DIM * DIM, scale))
        step_rms_a.append(_ones_vec(DIM))
        step_rms_b.append(_ones_vec(DIM))
        # gate/up: [HDIM * DIM], down: [DIM * HDIM]
        step_gate_a.append(_rand_vec(HDIM * DIM, scale))
        step_up_a.append(_rand_vec(HDIM * DIM, scale))
        step_down_a.append(_rand_vec(DIM * HDIM, scale))
        step_gate_b.append(_rand_vec(HDIM * DIM, scale))
        step_up_b.append(_rand_vec(HDIM * DIM, scale))
        step_down_b.append(_rand_vec(DIM * HDIM, scale))


# ===================================================================
# BLEND (calendar-driven dual matrix interpolation)
# ===================================================================
def get_blend_alpha():
    cd = calendar_dissonance(calendar_days_since_epoch())
    pd = personal_dissonance()
    return max(0.0, min(1.0, 0.5 + 0.3 * (cd - 0.5) - 0.2 * prophecy_debt + 0.1 * pd))


# ===================================================================
# CO-OCCURRENCE + BIGRAMS
# ===================================================================
cooc = {}
bigrams = {}


def update_cooc(w1, w2):
    key = (min(w1, w2), max(w1, w2))
    cooc[key] = cooc.get(key, 0) + 1


def get_cooc(w1, w2):
    return cooc.get((min(w1, w2), max(w1, w2)), 0)


def update_bigram(prev, nxt):
    key = (prev, nxt)
    bigrams[key] = bigrams.get(key, 0) + 1


def get_bigram(prev, nxt):
    return bigrams.get((prev, nxt), 0)


# ===================================================================
# FORWARD STEP -- pool_context -> Wr -> RMSNorm -> SwiGLU -> logits
# Architecture from penelope.c, adapted for Janus dual matrices
# ===================================================================

def silu(x):
    return x / (1.0 + math.exp(-x)) if x > -20 else 0.0


def forward_step(bpe_ids, step_idx):
    """RRPRAM forward: BPE input -> blended embed_in -> Wr -> RMSNorm -> SwiGLU -> logits(embed_out)."""
    s = step_idx % STEPS
    a = get_blend_alpha()
    b = 1.0 - a

    # Pool context: average blended embed_in over all BPE tokens
    ctx = [0.0] * DIM
    n = len(bpe_ids) if bpe_ids else 1
    for tok_id in bpe_ids:
        if tok_id >= BPE_VOCAB:
            continue
        base = tok_id * DIM
        for d in range(DIM):
            ctx[d] += a * embed_in_A[base + d] + b * embed_in_B[base + d]
    inv = 1.0 / n
    for d in range(DIM):
        ctx[d] *= inv

    # RRPRAM: query = ctx @ Wr (blended)
    query = [0.0] * DIM
    for i in range(DIM):
        acc = 0.0
        for j in range(DIM):
            acc += (a * step_wr_a[s][i * DIM + j] + b * step_wr_b[s][i * DIM + j]) * ctx[j]
        query[i] = acc

    # RMSNorm
    ss = sum(q * q for q in query)
    rms_inv = 1.0 / math.sqrt(ss / DIM + 1e-5)
    qn = [(a * step_rms_a[s][i] + b * step_rms_b[s][i]) * query[i] * rms_inv
          for i in range(DIM)]

    # SwiGLU: gate[HDIM] = W_gate[HDIM,DIM] @ qn[DIM]
    gate = [0.0] * HDIM
    up = [0.0] * HDIM
    for i in range(HDIM):
        sg = 0.0
        su = 0.0
        for j in range(DIM):
            sg += (a * step_gate_a[s][i * DIM + j] + b * step_gate_b[s][i * DIM + j]) * qn[j]
            su += (a * step_up_a[s][i * DIM + j] + b * step_up_b[s][i * DIM + j]) * qn[j]
        gate[i] = sg
        up[i] = su

    swiglu = [silu(gate[i]) * up[i] for i in range(HDIM)]

    # Down: hidden[DIM] = W_down[DIM,HDIM] @ swiglu[HDIM]
    hidden = [0.0] * DIM
    for i in range(DIM):
        acc = 0.0
        for j in range(HDIM):
            acc += (a * step_down_a[s][i * HDIM + j] + b * step_down_b[s][i * HDIM + j]) * swiglu[j]
        hidden[i] = acc

    # Residual
    out = [qn[i] + hidden[i] for i in range(DIM)]

    # Logits = embed_out @ out (separate output embeddings, no weight tying)
    logits = [0.0] * NWORDS
    for v in range(NWORDS):
        acc = 0.0
        vbase = v * DIM
        for d in range(DIM):
            acc += (a * embed_out_A[vbase + d] + b * embed_out_B[vbase + d]) * out[d]
        logits[v] = acc

    return logits


# ===================================================================
# DARIO OVERLAY -- heuristic forces applied on top of learned logits
# ===================================================================

def dario_overlay(logits, word_context, prev_word, step_idx, direction):
    """Apply Hebbian co-occurrence, Prophecy, Destiny overlay on top of learned logits."""
    alpha_mod = 1.0 + 0.3 * chambers[CH_LOVE] - 0.2 * chambers[CH_RAGE] + 0.1 * chambers[CH_FLOW]
    gamma_mod = 1.0 + 0.4 * chambers[CH_VOID] + 0.2 * chambers[CH_COMPLEX]
    cal_mod = 1.0 + 0.2 * calendar_dissonance(calendar_days_since_epoch())
    dir_mod = 0.8 if direction == -1 else 1.2
    gate_r = 1.0 / (1.0 + math.exp(-(resonance_field - 0.5) * 4))
    h_g = silu(gate_r * 2)
    f_g = silu(gate_r * 1.5)

    for v in range(NWORDS):
        # B: bigram transition
        B = 0.0
        if prev_word >= 0:
            B = math.log(1 + get_bigram(prev_word, v)) * 4

        # H: Hebbian co-occurrence
        H = 0.0
        for c in word_context:
            H += math.log(1 + get_cooc(c, v))
        H /= (len(word_context) + 1)

        # F: Prophecy fulfillment (scaled by debt)
        F = prophecy_debt * (1.0 + random.random() * 0.5)

        # A: Destiny attraction
        A = destiny_bias * gamma_mod * 0.5

        logits[v] += (B + alpha_mod * 3 * H * h_g + 2 * F * f_g + A) * dir_mod * cal_mod


# ===================================================================
# WORD SELECTION -- forward_step + dario overlay + top-k sampling
# ===================================================================

def select_word(bpe_context, word_context, prev_word, step_idx, forbidden, direction):
    """Run RRPRAM forward step, apply Dario overlay, sample from top-k."""
    logits = forward_step(bpe_context, step_idx)
    dario_overlay(logits, word_context, prev_word, step_idx, direction)

    # Mask forbidden words
    for f in forbidden:
        logits[f] = -1e9

    # Build scored candidates
    indexed = [{'word': WORDS[i], 'idx': i, 'score': logits[i]} for i in range(NWORDS)]
    indexed.sort(key=lambda s: s['score'], reverse=True)

    top_k = min(8, len(indexed))
    top = indexed[:top_k]

    # Softmax sampling over top-k
    mx = top[0]['score']
    for s in top:
        s['prob'] = math.exp(s['score'] - mx)
    total = sum(s['prob'] for s in top)
    for s in top:
        s['prob'] /= total

    r = random.random()
    cum = 0.0
    for s in top:
        cum += s['prob']
        if cum >= r:
            return s
    return top[0]


# ===================================================================
# ORIGIN EXTRACTION (find most "charged" word in prompt)
# ===================================================================

def extract_key(text):
    ids = tokenize_words(text)
    if not ids:
        return random.randint(0, NWORDS - 1)
    best = ids[0]
    best_score = 0
    for wid in ids:
        sc = 0
        for key, val in cooc.items():
            if key[0] == wid or key[1] == wid:
                sc += val
        if sc > best_score:
            best_score = sc
            best = wid
    return best


# ===================================================================
# BI-DIRECTIONAL REASONING (simultaneous forward + backward)
# ===================================================================

def run_chain(user_text):
    """Run the 12-step bi-directional generation chain.

    Matches the HTML version: origin in the center, backward words grow
    upward and forward words grow downward, interleaved simultaneously.
    """
    global prophecy_debt, chambers

    # BPE encode user input for forward step context
    input_bpe = bpe_encode(user_text)
    # Word-level tokenize for Dario overlay / co-occurrence
    word_context = tokenize_words(user_text)
    seed_idx = extract_key(user_text)
    seed_word = WORDS[seed_idx] if seed_idx < NWORDS else 'void'

    # MetaJanus prophecy
    cal_d = calendar_dissonance(calendar_days_since_epoch())
    predicted_entropy = 0.5 + 0.2 * prophecy_debt + 0.1 * cal_d + 0.15 * personal_dissonance()

    # Step direction split
    n_backward = max(1, min(STEPS - 1,
                            int(STEPS * (0.3 + 0.4 * prophecy_debt + 0.1 * cal_d))))
    n_forward = STEPS - n_backward

    forbidden = set()
    all_steps = []

    # Two independent contexts growing simultaneously
    fwd_bpe_ctx = list(input_bpe)
    fwd_word_ctx = list(word_context)
    fwd_prev = seed_idx

    bwd_bpe_ctx = list(input_bpe)
    bwd_word_ctx = list(word_context)
    bwd_prev = seed_idx

    back_steps = []
    fwd_steps = []

    max_steps = max(n_forward, n_backward)

    for i in range(max_steps):
        # Generate BACKWARD word (grows upward from origin)
        if i < n_backward:
            update_chambers(n_forward + i)
            sel = select_word(bwd_bpe_ctx, bwd_word_ctx, bwd_prev,
                              n_forward + i, forbidden, -1)
            step_info = {
                'word': sel['word'], 'idx': sel['idx'],
                'step': n_forward + i,
                'wormhole': False, 'debt': prophecy_debt,
            }
            back_steps.append(step_info)
            all_steps.append(sel)

            # Extend backward BPE context with selected word's BPE tokens
            wb = vocab_bpe[sel['idx']]
            if wb:
                bwd_bpe_ctx.extend(wb)
            bwd_word_ctx.append(sel['idx'])
            forbidden.add(sel['idx'])
            prophecy_debt = 0.9 * prophecy_debt + 0.1 * compute_prophecy_debt([sel], 0)
            bwd_prev = sel['idx']

        # Generate FORWARD word (grows downward from origin)
        if i < n_forward:
            update_chambers(i)
            is_wormhole = False
            if prophecy_debt < 0.2 and wormhole > 0.1 and random.random() < wormhole:
                is_wormhole = True

            sel = select_word(fwd_bpe_ctx, fwd_word_ctx, fwd_prev,
                              i, forbidden, 1)
            step_info = {
                'word': sel['word'], 'idx': sel['idx'], 'step': i,
                'wormhole': is_wormhole, 'debt': prophecy_debt,
            }
            fwd_steps.append(step_info)
            all_steps.append(sel)

            # Extend forward BPE context
            wb = vocab_bpe[sel['idx']]
            if wb:
                fwd_bpe_ctx.extend(wb)
            fwd_word_ctx.append(sel['idx'])

            if fwd_prev >= 0:
                update_bigram(fwd_prev, sel['idx'])
            for c in word_context:
                update_cooc(c, sel['idx'])
            forbidden.add(sel['idx'])
            prophecy_debt = 0.9 * prophecy_debt + 0.1 * compute_prophecy_debt([sel], 0)
            fwd_prev = sel['idx']

    # Evaluate MetaJanus prophecy
    avg_debt = sum(x.get('debt', 0) for x in all_steps) / len(all_steps) if all_steps else 0
    error = abs(predicted_entropy - avg_debt)
    META.prophecy_accuracy = 0.9 * META.prophecy_accuracy + 0.1 * (1 - error)
    META.total_predictions += 1

    return seed_word, seed_idx, fwd_steps, back_steps


def display_chain(seed_word, fwd_steps, back_steps):
    """Print the bi-directional chain to the terminal.

    Layout matches HTML: backward steps on top (reversed), origin in center,
    forward steps below.
    """
    # Backward steps (top, reversed -- most recent backward at top)
    for s in reversed(back_steps):
        wh = ' +WH' if s['wormhole'] else ''
        print(f"  ^{s['step']:<3} {s['word']:<16} debt={s['debt']:.2f}{wh}")

    # Origin
    print(f"  * {seed_word} *".center(40) + '  [ORIGIN]')

    # Forward steps (bottom)
    for s in fwd_steps:
        wh = ' +WH' if s['wormhole'] else ''
        print(f"  v{s['step']:<3} {s['word']:<16} debt={s['debt']:.2f}{wh}")


def display_metrics():
    """Print drift/dissonance/blend/debt metrics."""
    days = calendar_days_since_epoch()
    drift = calendar_cumulative_drift(days)
    diss = calendar_dissonance(days)
    pd = personal_dissonance()
    blend = get_blend_alpha()
    print(f"\ndrift={drift:.2f}  diss={diss:.3f}  personal={pd:.3f}  "
          f"blend={blend:.3f}  debt={prophecy_debt:.3f}  "
          f"prophecy_acc={META.prophecy_accuracy:.3f}  "
          f"bpe={BPE_VOCAB}  words={NWORDS}")


# ===================================================================
# TRAINING -- BPE input -> word prediction with RRPRAM forward + Chuck
# ===================================================================

def train_on_text(text, total_steps=2000, lr=0.001):
    """Training loop: BPE input embeddings, word output embeddings,
    RRPRAM per-step weights. Chuck optimizer (macro patience, lambda
    modulation, stagnation noise)."""
    global embed_in_A, embed_in_B, embed_out_A, embed_out_B

    # Split text into words, find those in vocabulary
    raw_words = re.sub(r'[^a-z\s]', '', text.lower()).split()
    word_ids = []
    for w in raw_words:
        if w in WORDS:
            word_ids.append(WORDS.index(w))

    if len(word_ids) < STEPS + 2:
        print(f'Error: too few vocabulary words found in text '
              f'({len(word_ids)} words, need at least {STEPS + 2})')
        return

    print(f'Tokenized: {len(word_ids)} vocab words found')
    print(f'Training for {total_steps} steps ...')

    # Chuck optimizer state
    best_loss = float('inf')
    patience_counter = 0
    macro_patience = 200
    chuck_lambda = 1.0

    for step in range(total_steps):
        offset = random.randint(0, len(word_ids) - STEPS - 2)
        loss = 0.0
        a = get_blend_alpha()

        for si in range(STEPS):
            # Build BPE context from preceding words
            bpe_ctx = []
            for k in range(offset, offset + si + 1):
                wb = vocab_bpe[word_ids[k]]
                if wb:
                    bpe_ctx.extend(wb)
            target = word_ids[offset + si + 1]

            # Pool embed_in_A
            ctx = [0.0] * DIM
            n = len(bpe_ctx) if bpe_ctx else 1
            for tok_id in bpe_ctx:
                if tok_id >= BPE_VOCAB:
                    continue
                base = tok_id * DIM
                for d in range(DIM):
                    ctx[d] += embed_in_A[base + d]
            inv = 1.0 / n
            for d in range(DIM):
                ctx[d] *= inv

            # RRPRAM: query = ctx @ Wr
            query = [0.0] * DIM
            for i in range(DIM):
                acc = 0.0
                for j in range(DIM):
                    acc += step_wr_a[si][i * DIM + j] * ctx[j]
                query[i] = acc

            # RMSNorm
            ss = sum(q * q for q in query)
            rms_inv = 1.0 / math.sqrt(ss / DIM + 1e-5)
            qn = [step_rms_a[si][i] * query[i] * rms_inv for i in range(DIM)]

            # SwiGLU forward
            gate = [0.0] * HDIM
            up = [0.0] * HDIM
            for i in range(HDIM):
                sg = 0.0
                su = 0.0
                for j in range(DIM):
                    sg += step_gate_a[si][i * DIM + j] * qn[j]
                    su += step_up_a[si][i * DIM + j] * qn[j]
                gate[i] = sg
                up[i] = su
            swiglu = [silu(gate[i]) * up[i] for i in range(HDIM)]

            hidden = [0.0] * DIM
            for i in range(DIM):
                acc = 0.0
                for j in range(HDIM):
                    acc += step_down_a[si][i * HDIM + j] * swiglu[j]
                hidden[i] = acc

            out = [qn[i] + hidden[i] for i in range(DIM)]

            # Logits = embed_out_A @ out
            logits = [0.0] * NWORDS
            for v in range(NWORDS):
                acc = 0.0
                vbase = v * DIM
                for d in range(DIM):
                    acc += embed_out_A[vbase + d] * out[d]
                logits[v] = acc

            # Softmax + cross-entropy loss
            mx = max(logits)
            exp_logits = [math.exp(l - mx) for l in logits]
            total = sum(exp_logits)
            probs = [e / total for e in exp_logits]

            p = max(1e-10, probs[target])
            loss -= math.log(p)

            # Gradient: d_logits = probs - one_hot
            d_logits = list(probs)
            d_logits[target] -= 1.0

            chuck_mod = chuck_lambda

            # Update embed_out_A
            for v in range(NWORDS):
                if abs(d_logits[v]) < 1e-6:
                    continue
                vbase = v * DIM
                for d in range(DIM):
                    embed_out_A[vbase + d] -= lr * chuck_mod * d_logits[v] * out[d]

            # Update embed_in_A (gradient through pool_context)
            for tok_id in bpe_ctx:
                if tok_id >= BPE_VOCAB:
                    continue
                base = tok_id * DIM
                for d in range(DIM):
                    embed_in_A[base + d] -= lr * chuck_mod * 0.01 * qn[d] / n

            # Update Wr
            for i in range(DIM):
                for j in range(DIM):
                    step_wr_a[si][i * DIM + j] -= lr * chuck_mod * 0.01 * qn[i] * ctx[j]

        # Train matrix B on even steps (simpler: next-word embedding similarity)
        if step % 2 == 0:
            off2 = random.randint(0, len(word_ids) - 2)
            src_word = word_ids[off2]
            tgt_word = word_ids[off2 + 1]
            src_bpe = vocab_bpe[src_word]
            if src_bpe and len(src_bpe) > 0:
                ctx2 = [0.0] * DIM
                for tok_id in src_bpe:
                    if tok_id >= BPE_VOCAB:
                        continue
                    base = tok_id * DIM
                    for d in range(DIM):
                        ctx2[d] += embed_in_B[base + d]
                inv2 = 1.0 / len(src_bpe)
                for d in range(DIM):
                    ctx2[d] *= inv2
                for v in range(NWORDS):
                    sc = 0.0
                    vbase = v * DIM
                    for d in range(DIM):
                        sc += embed_out_B[vbase + d] * ctx2[d]
                    grad = (-1.0 if v == tgt_word else 0.0) + 0.001
                    for d in range(DIM):
                        embed_out_B[vbase + d] -= lr * grad * ctx2[d] * 0.1

        # Chuck optimizer: macro patience + stagnation noise
        avg_loss = loss / STEPS
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            chuck_lambda = max(0.5, chuck_lambda * 0.99)
        else:
            patience_counter += 1
            if patience_counter > macro_patience:
                chuck_lambda = min(2.0, chuck_lambda * 1.05)
                noise_scale = 0.001 * chuck_lambda
                for idx in range(NWORDS * DIM):
                    embed_out_A[idx] += (random.random() - 0.5) * noise_scale
                patience_counter = 0

        if step % 100 == 0:
            print(f'  step {step:>5}/{total_steps}  loss={avg_loss:.4f}  '
                  f'chuck_l={chuck_lambda:.3f}  best={best_loss:.4f}')

    print(f'Training complete: {total_steps} steps, final best loss={best_loss:.4f}')


# ===================================================================
# SAVE / LOAD WEIGHTS (pickle)
# ===================================================================
WEIGHTS_FILE = 'nanojanus.weights.pkl'


def save_weights(path=WEIGHTS_FILE):
    data = {
        'embed_in_A': embed_in_A, 'embed_in_B': embed_in_B,
        'embed_out_A': embed_out_A, 'embed_out_B': embed_out_B,
        'step_wr_a': step_wr_a, 'step_wr_b': step_wr_b,
        'step_rms_a': step_rms_a, 'step_rms_b': step_rms_b,
        'step_gate_a': step_gate_a, 'step_up_a': step_up_a,
        'step_down_a': step_down_a,
        'step_gate_b': step_gate_b, 'step_up_b': step_up_b,
        'step_down_b': step_down_b,
        'cooc': cooc, 'bigrams': bigrams,
        'prophecy_debt': prophecy_debt,
        'meta': {
            'prophecy_accuracy': META.prophecy_accuracy,
            'total_predictions': META.total_predictions,
        },
        'nwords': NWORDS,
        'bpe_vocab': BPE_VOCAB,
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Weights saved to {path}')


def load_weights(path=WEIGHTS_FILE):
    global embed_in_A, embed_in_B, embed_out_A, embed_out_B
    global prophecy_debt, cooc, bigrams
    global step_wr_a, step_wr_b, step_rms_a, step_rms_b
    global step_gate_a, step_up_a, step_down_a
    global step_gate_b, step_up_b, step_down_b

    if not os.path.exists(path):
        return False
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if data.get('nwords') != NWORDS:
        print(f'Warning: weight file vocab size ({data.get("nwords")}) '
              f'!= current ({NWORDS}), reinitializing')
        return False
    if data.get('bpe_vocab', BPE_VOCAB) != BPE_VOCAB:
        print(f'Warning: weight file BPE vocab ({data.get("bpe_vocab")}) '
              f'!= current ({BPE_VOCAB}), reinitializing')
        return False

    embed_in_A = data['embed_in_A']
    embed_in_B = data['embed_in_B']
    embed_out_A = data['embed_out_A']
    embed_out_B = data['embed_out_B']
    step_wr_a = data['step_wr_a']
    step_wr_b = data['step_wr_b']
    step_rms_a = data['step_rms_a']
    step_rms_b = data['step_rms_b']
    step_gate_a = data['step_gate_a']
    step_up_a = data['step_up_a']
    step_down_a = data['step_down_a']
    step_gate_b = data['step_gate_b']
    step_up_b = data['step_up_b']
    step_down_b = data['step_down_b']
    cooc.update(data.get('cooc', {}))
    bigrams.update(data.get('bigrams', {}))
    prophecy_debt = data.get('prophecy_debt', 0.0)
    meta = data.get('meta', {})
    META.prophecy_accuracy = meta.get('prophecy_accuracy', 0.5)
    META.total_predictions = meta.get('total_predictions', 0)
    print(f'Weights loaded from {path}')
    return True


# ===================================================================
# CLI
# ===================================================================
def main():
    global META

    parser = argparse.ArgumentParser(
        description='NanoJanus -- Bi-directional associative reasoning '
                    '(real BPE dual tokenizer architecture)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--generate', type=str, metavar='PROMPT',
                       help='Generate a bi-directional chain from prompt text')
    group.add_argument('--train', type=str, metavar='FILE',
                       help='Train on a text file')
    parser.add_argument('--vocab', type=str, default=None,
                        help='Path to vocabulary file (default: nanojanus.txt beside script)')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Training steps (default: 2000)')
    parser.add_argument('--weights', type=str, default=WEIGHTS_FILE,
                        help='Weights file path (default: nanojanus.weights.pkl)')
    args = parser.parse_args()

    # Locate vocabulary file
    vocab_path = args.vocab
    if vocab_path is None:
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'nanojanus.txt')
    if not os.path.exists(vocab_path):
        print(f'Error: vocabulary file not found: {vocab_path}', file=sys.stderr)
        sys.exit(1)

    load_vocabulary(vocab_path)
    META = MetaJanus()
    init_weights()
    load_weights(args.weights)

    print(f'[nanojanus] ready: {NWORDS} words, BPE vocab {BPE_VOCAB}, '
          f'{BPE_MERGES} merges, {len(vocab_bpe)} words BPE-precomputed')

    if args.generate:
        seed_word, seed_idx, fwd_steps, back_steps = run_chain(args.generate)
        print()
        display_chain(seed_word, fwd_steps, back_steps)
        display_metrics()
        print()

    elif args.train:
        if not os.path.exists(args.train):
            print(f'Error: training file not found: {args.train}', file=sys.stderr)
            sys.exit(1)
        with open(args.train, 'r') as f:
            text = f.read()
        train_on_text(text, total_steps=args.steps)
        save_weights(args.weights)


if __name__ == '__main__':
    main()
