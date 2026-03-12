#let table_underline(s) = [
  #set text(baseline: 5pt)
  #s
  #v(-0.5em)
  #line(length: 100%, stroke: 1pt)
]

#let justify(s) = {
  set text(weight: "bold")
  if type(s) == content and s.has("text") { s = s.text }
  assert(type(s) == type("string"))
  s.clusters().join(h(1fr))
}

#let cover(
  title: "论文题目",
  course: "课程名称",
  class: "班级",
  student-id: "学号",
  student-name: "姓名",
) = {
  set page(paper: "a4", margin: 2cm)
  set text(14pt, font: "SimSun")
  grid(
    columns: (auto, 1fr),
    align: (left, right),
    image("cau-logo.png", width: 100pt),
    table(
      columns: 80pt,
      rows: (25pt, 25pt),
      align: center + horizon,
      [成绩],
      [],
    ),
  )
  v(8pt)

  align(center)[
    #text(size: 36pt)[
      中国农业大学\
      课程实验报告
    ]
  ]

  v(100pt)
  align(center)[
    #box(width: 80%)[
      #set text(16pt)
      #table(
        columns: (120pt, 2pt, 1fr),
        rows: 40pt,
        align: center + bottom,
        stroke: none,
        justify[实验题目], [:], table_underline[#title],
        justify[课程名称], [:], table_underline[#course],
        justify[班级], [:], table_underline[#class],
        justify[学号], [:], table_underline[#student-id],
        justify[姓名], [:], table_underline[#student-name],
      )
    ]
  ]
}


// #cover(
//   title: "论新时期中美两国正确相处之道：从竞争博弈到共生共荣",
//   course: "形式与政策",
//   class: "计算231",
//   student-id: "202330825011",
//   student-name: "龚浩然",
// )
