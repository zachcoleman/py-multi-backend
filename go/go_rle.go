package go_rle

func MaskToRle(arr []bool) []int {
	counts := []int{}
	count := 0
	tgt := false

	for _, curr := range arr {
		if curr == tgt {
			count++
		} else {
			counts = append(counts, count)
			tgt = curr
			count = 1
		}
	}
	counts = append(counts, count)
	return counts
}
