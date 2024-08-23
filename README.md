# Spam Movie Review Detection

This is the official repository of paper "Detecting Spam Movie Review under Coordinated Attack with Multi-View Explicit and Implicit Relations Semantics Fusion", which is published in the journal *IEEE Transactions on Information Forensics and Security (TIFS, 2024)*.

## Code

See folder `Code/`. *Note that the original version of code is runned on our self-developed deep-learning experimental platform.* Contact Yicheng at cyc.ralion@gmail.com for any questions.

## Dataset

The dataset used in this paper includes two parts: 

- The first part is a target review system dataset, which includes review dataset and movie dataset from [Maoyan](https://www.maoyan.com/) movie review platform.
- The second part is external review system dataset, which includes professional reviews from other popular movie review platforms. Specifically, we constructed two datasets from [Douban](https://movie.douban.com/) and [Mtime](http://www.mtime.com/).

### How to download

The dataset is available at [Google Drive](https://drive.google.com/drive/folders/1dbw3pl4qy9W6kzXA2upeTba5ObLUwzmb?usp=drive_link).

Please apply for access by contacting Yicheng at cyc.ralion@gmail.com with your institutional email address and clearly state your institution, your research advisor (if any), and your use case of our dataset.

### Data Format and Sample

- The detailed description of fields in review dataset from target review system (Maoyan) are as follows:

| Field Name            | Description                                                              | Example                            |
| --------------------- | ------------------------------------------------------------------------ | ---------------------------------- |
| Label                 | Label of review (1 for spam, 0 for non-spam)                             | 1                                  |
| Content               | Content of review                                                        | 懒得评论，我的评论瞎说，请勿模仿！ |
| Score                 | Score of review                                                          | 8                                  |
| Thumb-up              | Thumb-ups number of review                                               | 0                                  |
| Comment               | Comments number under the review                                         | 0                                  |
| Published Time        | Published time of review                                                 | 2021-11-06 12:12:00                |
| Nickname              | Username of user on Maoyan platform                                      | 承诺是神话                         |
| User ID               | User id on Maoyan platform                                               | 253481540                          |
| Bought Ticket         | Whether user has bought a ticket of this movie on Maoyan Platform or not | FALSE                              |
| User Rank             | Rank of Maoyan user account                                              | 2                                  |
| User Watched Number   | Number of movies user watched                                            | 10                                 |
| User To-Watch Number  | Number of movies user want to watch                                      | 10                                 |
| User Review Number    | Number of movie reviews user wrote                                       | 2                                  |
| User Topic Number     | Number of topics user is interested in                                   | 0                                  |
| Movie ID              | Movie id on Maoyan platform                                              | 1337700                            |
<!-- | Movie Year            | Published year of movie                                                  | 2021                               |
| Movie Name            | Name of Movie                                                            | 中国医生                           |
| Movie Published Date  | Published date of movie                                                  | 2021-07-09 00:00:00                |
| High Score Rate       | Percentage of users who score the movie from 9 to 10                     | 88.87                              |
| Medium Score Rate     | Percentage of users who score the movie from 5 to 8                      | 8.08                               |
| Low Score Rate        | Percentage of users who score the movie from 1 to 4                      | 3.05                               |
| Movie To-Watch Number | Number of users who want to watch the movie                              | 475631                             |
| Movie Watched Number  | Number of users who watched the movie                                    | 36540296                           |
| Movie Genre           | Genres of movie (delimted by comma)                                                         | 剧情 | -->

- The detailed description of fields in movie dataset from target review system (Maoyan) are as follows:

| Field Name        | Description                                          | Example             |
| ----------------- | ---------------------------------------------------- | ------------------- |
| ID                | Movie id on Maoyan platform                          | 1337700             |
| Year              | Published year of movie                              | 2021                |
| Name              | Name of Movie                                        | 中国医生            |
| Published Date    | Published date of movie                              | 2021-07-09 00:00:00 |
| High Score Rate   | Percentage of users who score the movie from 9 to 10 | 88.87               |
| Medium Score Rate | Percentage of users who score the movie from 5 to 8  | 8.08                |
| Low Score Rate    | Percentage of users who score the movie from 1 to 4  | 3.05                |
| To-Watch Number   | Number of users who want to watch the movie          | 475631              |
| Watched Number    | Number of users who watched the movie                | 36540296            |
| Genre             | Genres of movie (delimted by comma)                  | 剧情                |
|Synopsis|Synopsis of movie|电影《中国医生》根据新冠肺炎疫情防控斗争的真实事件改编，以武汉市金银潭医院为核心故事背景，同时兼顾武汉同济医院、武汉市肺科医院、武汉协和医院、武汉大学人民医院（湖北省人民医院）、火神山医院、方舱医院等兄弟单位，以武汉医护人员、全国各省市援鄂医疗队为人物原型，全景式记录波澜壮阔、艰苦卓绝的抗疫斗争。|

- The detailed description of fields in external review dataset from Douban are as follows:

| Field Name      | Description                                           | Example                                                                                                                                                                            |
| --------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ID              | Movie id on Douban platform                           | 25858758                                                                                                                                                                           |
| Maoyan ID       | Movie id on Maoyan platform                           | 1337700                                                                                                                                                                            |
| Name            | Name of movie                                         | 中国医生                                                                                                                                                                           |
| User ID         | User id on Douban platform                            | 28165185                                                                                                                                                                           |
| Nickname        | Username of user on Douban platform                   | 林微云                                                                                                                                                                             |
| Content         | Content of review                                     | 这部电影很多场景真实得让人不忍直视，有时候甚至让人恍惚中觉得这不像是电影，更像是某种意义上的纪录片，让人不自觉地想起了疫情期间的回忆。**TLDR.** 感谢奋战在抗疫一线的所有医务人员！ |
| Score           | Score of review                                       | 4                                                                                                                                                                                  |
| Published Time  | Published time of review                              | 2021-07-09 15:38:37                                                                                                                                                                |
| Useful  Number  | Number of users who consider the review to be useful  | 109                                                                                                                                                                                |
| Useless  Number | Number of users who consider the review to be useless | 14                                                                                                                                                                                 |
| Comment  Number | Number of users who commented under the review        | 19 |


- The detailed description of fields in external review dataset from Mtime are as follows:

| Field Name      | Description                                           | Example                                                                                                                                                                            |
| --------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ID              | Movie id on Mtime platform                           | 268794                                                                                                                                                                           |
| Maoyan ID       | Movie id on Maoyan platform                           | 1337700                                                                                                                                                                            |
| Name            | Name of movie                                         | 中国医生                                                                                                                                                                           |
| User ID         | User id on Mtime platform                            | 781648                                                                                                                                                                           |
| Nickname        | Username of user on Mtime platform                   | 莫选好片                                                                                                                                                                             |
| Content         | Content of review                                     | 博纳拍主旋律电影已经逐渐找准了方向，从“中国骄傲三部曲”到“中国胜利三部曲”，不但票房取得一个又一个成功，也填补了相关题材的空白。特别是这次的《中国医生》，​ **TLDR.** 尽管有着诸多问题，可到了最后，苏莫还是推荐这部电影的。因为，题材本身已经赢了。 |
| Score           | Score of review                                       | 6                                                                                                                                                                                  |
| Published Time  | Published time of review                              | 2021-07-10 12:30:52                                                                                                                                                                |
| Useful  Number  | Number of users who consider the review to be useful  | 8                                                                                                                                                                                |
| Useless  Number | Number of users who consider the review to be useless | 2                                                                                                                                                                                 |
| Comment  Number | Number of users who commented under the review        | 3 |


## Cite (BibTex)

Please cite the following paper, if you find our work useful in your research:

```
@ARTICLE{cai2024detecting,
  author={Cai, Yicheng and Wang, Haizhou and Cao, Hao and Wang, Wenxian and Zhang, Lei and Chen, Xingshu},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Detecting Spam Movie Review Under Coordinated Attack With Multi-View Explicit and Implicit Relations Semantics Fusion}, 
  year={2024},
  volume={19},
  number={},
  pages={7588-7603},
  doi={10.1109/TIFS.2024.3441947}}
```
