We collected a specific subset form VisDial v1.0 val, in which questions contain high-frequency words “yes” or “no”.  VisDial v1.0 val set contains 2,064 dialog samples with 20,640 questions. We selected the questions whose ground-truth answer is “yes” or “no”, giving us with 8,060 questions. We then further removed the samples that do not include “yes” or “no” in the dialog history and the samples in which the word frequencies of “yes” and “no” are consistent. Finally, we acquired 1,729 samples with a total of 6,778 questions.  We call this subset of VisDial v1.0 val as VisDial v1.0 (val-yn).

The data format of VisDial v1.0 (val-yn) is as follows:

```
{
  'data': {
    'questions': [
      'does it have a doorknob',
      'do you see a fence around the bear',
      ...
    ],
    'answers': [
      'no, there is just green field in foreground',
      'countryside house',
      ...
    ],
    'dialogs': [
      {
        'image_id': <image id>,
        'caption': <image caption>,
        'dialog': [
          {
            'question': <index of question in `data.questions` list>,
            'answer': <index of answer in `data.answers` list>,
            'answer_options': <100 candidate answer indices from `data.answers`>,
            'gt_index': <index of `answer` in `answer_options`>
          },
          ... (10 rounds of dialog)
        ],
        'round_id' [<index of question that will be evaluated>]
      },
      ...
    ]
  },
  'split': val_yn,
  'version': '1.0'
}
```

