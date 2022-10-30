1. 题目如下，要求实现SnapshotSet Interface，能够用iterator()取某一时刻这个set的所有元素，
在调用iterator()和使用Iterator<T>之间，可以对SnapshotSet进行修改，不考虑多线程，所有操作都是顺序执行:
// java
// interface SnapshotSet<T> {
//       void add(T e);
//       void remove(T e);
//       boolean contains(T e);
//       Iterator<T> iterator(); // iterator() should return a snapshot of the elements in the collection at the time the method was called.
// }
例子:
// Add | Remove
// 5 |
// 2 |
// 8 |
//   | 5
// --------- it <- iterator() (iterator only created, not used yet)，此时it里有2和8。
// 1 |
// ----‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌----- contains(2) = true
//   | 2
// --------- contains(2) = false
// --------- [x for x in it] = [2, 8] (No ordering guarantees)，把2和8读出来。
// # https://www.1point3acres.com/bbs/thread-873494-1-1.html

"""
class ValueNotExistException(message: String) : Exception(message)

class SnapshotIteratorNoSuchElementException(message: String) : Exception(message)

interface SnapshotSet<T> {
    fun add(value: T)

    fun remove(value: T)

    fun contains(value: T): Boolean

    fun iterator(): Iterator<T>
}

// use a list to store the actions history of each value
// use a map to map the value in set to the index in snapshot list
// store a value as current snapshot id
// when there is add/remove increase the snapshot id and insert the action for certain value
// for snapshot iterator, use index mover iterator through the snapshot list to find presented value
// whose action snapshot id is smaller than passed in snapshot id
class SimpleSnapShotSet<T> : SnapshotSet<T> {
    private val internalSet = mutableSetOf<T>()
    private val valueToSnapshotIndexMap = mutableMapOf<T, Int>()
    private val snapshots = mutableListOf<MutableList<Action<T>>>()
    private var curSnapshotId = 0

    override fun add(value: T) {
        internalSet.add(value)

        curSnapshotId++
        if (valueToSnapshotIndexMap.contains(value)) {
            val snapshotIndex = valueToSnapshotIndexMap[value]!!
            snapshots[snapshotIndex].add(Action(curSnapshotId, true, value))
        } else {
            val index = snapshots.size
            snapshots.add(mutableListOf())
            snapshots[index].add(Action(curSnapshotId, true, value))
            valueToSnapshotIndexMap[value] = index
        }
    }

    override fun remove(value: T) {
        if (!internalSet.contains(value)) {
            throw ValueNotExistException("try to delete value is not in set")
        }

        internalSet.remove(value)

        curSnapshotId++
        snapshots[valueToSnapshotIndexMap[value]!!].add(Action(curSnapshotId, false, value))
    }

    override fun contains(value: T): Boolean {
        return internalSet.contains(value)
    }

    override fun iterator(): Iterator<T> {
        return SnapshotIterator(snapshots, curSnapshotId)
    }
}

data class Action<T>(val snapshotId: Int, val isPresented: Boolean, val value: T)

class SnapshotIterator<T>(
        private val snapshots: List<List<Action<T>>>,
        private val snapshotId: Int
) : Iterator<T> {
    private var curIndex = -1

    override fun hasNext(): Boolean {
        // find next available value exists
        return findNextAvailableIndex() != null
    }

    override fun next(): T {
        // try to move to next available value
        curIndex = findNextAvailableIndex()
                ?: throw SnapshotIteratorNoSuchElementException("No element.")

        return snapshots[curIndex].first().value
    }

    private fun findNextAvailableIndex(): Int? {
        var mover = curIndex + 1

        while (mover < snapshots.size) {
            val actions = snapshots[mover]

            // find most close action
            val mostRecentAction = actions.filter { action ->
                action.snapshotId <= snapshotId
            }.lastOrNull()

            // if no action happen before current snapshot id or most recent one is deletion
            // then skip
            if (mostRecentAction == null || !mostRecentAction.isPresented) {
                mover++
                continue
            }

            return mover
        }

        return null
    }
}

"""




2. 题是地里经典面经题，customer->revenue  
那个题，要实现referral api，最后get api是要求给一个customer id，return他的总的revenue，已经top k revenue
// # https://www.1point3acres.com/bbs/thread-871941-1-1.html

"""
public class ConsumerRevenue {
    class ConsumerRevenuePair {
        public int id;
        public int rev;

        public ConsumerRevenuePair(int id, int rev) {
            this.id = id;
            this.rev = rev;
        }
    }

    private HashMap<Integer, Integer> idToRevenue;
    private TreeMap<Integer, Set<Integer>> revenueToIds;
    private int idCounter;

    public ConsumerRevenue() {
        this.idCounter = 0;
        this.idToRevenue = new HashMap<>();
        this.revenueToIds = new TreeMap<>();
    }

    // O(logN)
    public int insert(int revenue) {
        int newId = getNewId();

        idToRevenue.put(newId, revenue);
        revenueToIds.computeIfAbsent(revenue, key -> new HashSet<Integer>()).add(newId);

        return newId;
    }

    // O(logN)
    public int insert(int revenue, int referrerID) {
        if (!idToRevenue.containsKey(referrerID)) {
            throw new IllegalArgumentException("Referrer does not exists");
        }

        // update for referrer
        int curRevenue = idToRevenue.get(referrerID);
        idToRevenue.put(referrerID, revenue + curRevenue);

        revenueToIds.get(curRevenue).remove(referrerID);
        if (revenueToIds.get(curRevenue).isEmpty()) {
            revenueToIds.remove(curRevenue);
        }
        revenueToIds.computeIfAbsent(revenue + curRevenue, key -> new HashSet<Integer>()).add(referrerID);;

        // insert for new cx
        return insert(revenue);
    }

    // O(klogN)
    public List<Integer> getKLowestRevenue(int k, int targetRevenue) {
        System.out.println(revenueToIds);

        List<Integer> closetRevenueConsumerIds = new ArrayList<>();

        int nextTargetRevenue = targetRevenue;

        while (closetRevenueConsumerIds.size() < k) {
            Map.Entry<Integer, Set<Integer>> nextHigherRevenueEntry = revenueToIds.higherEntry(nextTargetRevenue);
            if (nextHigherRevenueEntry == null) {
                break;
            }

            Iterator<Integer> consumerIds = nextHigherRevenueEntry.getValue().iterator();
            while (closetRevenueConsumerIds.size() < k && consumerIds.hasNext()) {
                closetRevenueConsumerIds.add(consumerIds.next());
            }

            nextTargetRevenue = nextHigherRevenueEntry.getKey();
        }

        return closetRevenueConsumerIds;
    }

    public List<Integer> getKLowestRevenue2(int k, int targetRevenue) {
        List<Integer> result = new ArrayList<>();
        PriorityQueue<ConsumerRevenuePair> maxHeap = new PriorityQueue<>((o1, o2) -> o2.rev - o1.rev);

        int index = 0;
        List<Integer> ids = new ArrayList<>(idToRevenue.keySet());
        while (maxHeap.size() < k && index < ids.size()) {
            int curRev = idToRevenue.get(ids.get(index));
            if (curRev > targetRevenue) {
                System.out.println(curRev);

                maxHeap.offer(new ConsumerRevenuePair(ids.get(index), curRev));
            }

            index++;
        }

        System.out.println(maxHeap.peek().rev);

        while (index < ids.size()) {
            int curRev = idToRevenue.get(ids.get(index));
            if (curRev > targetRevenue && curRev < maxHeap.peek().rev) {
                maxHeap.poll();
                maxHeap.offer(new ConsumerRevenuePair(ids.get(index), curRev));
            }
            index++;
        }

        while (!maxHeap.isEmpty()) {
            result.add(maxHeap.poll().id);
        }

        return result;
    }


    public void print() {
        idToRevenue.keySet().forEach(key -> {
            System.out.println(key + "-" + idToRevenue.get(key));
        });
    }

    private int getNewId() {
        return idCounter++;
    }
}
"""

3. 
- 1. web crawler 问的比较细，主要是多线程，面试官还提示了可以优化的地方
- 2. 一堆数里面找topk，各种追问
- 3. 设计交易系统，之前面经里面有，一个烙印疯狂的挑毛病，体验极差‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌
// # https://www.1point3acres.com/bbs/thread-864726-1-1.html

"""
object CrawlerHelper {
    private val htmlMap = mapOf(
            "1" to "2/3/4",
            "2" to "3/4",
            "3" to "5",
            "4" to "5",
            "5" to "6"
    )

    fun fetch(url: String): HTML {
        return HTML(htmlMap[url] ?: "")
    }

    fun parse(html: HTML): List<String> {
        return html.html.split("/").filter { it.isNotEmpty() }
    }

    fun save(url: String, html: HTML) {
        Thread.sleep(1000)
        println("url $url is saved")
    }
}

// BFS through the urls
// the bottleneck is at I/O operations
// time complexity: O(N)
// space complexity: O(N)
class WebCrawler {
    fun crawl(startUrl: String, depth: Int) {
        val queue: Queue<UrlEntry> = LinkedList()
        val visited = mutableSetOf(startUrl)

        queue.offer(UrlEntry(startUrl, 1))

        while (queue.isNotEmpty()) {
            val curUrlEntry = queue.poll()

            // I/O blocking
            val curHtml = CrawlerHelper.fetch(curUrlEntry.url)
            CrawlerHelper.save(curUrlEntry.url, curHtml)

            if (curUrlEntry.depth >= depth) continue

            val nextUrls = CrawlerHelper.parse(curHtml)
            nextUrls.forEach { url ->
                if (!visited.contains(url)) {
                    visited.add(url)
                    queue.offer(UrlEntry(url, curUrlEntry.depth + 1))
                }
            }
        }

    }
}

data class UrlEntry(val url: String, val depth: Int)

data class HTML(val html: String)
"""

"""
/**
 * improve the efficiency by multi-thread
 * keys
 * 1 - avoid race condition where 2 threads check same url
 * and try to process as next url at same time (this causes duplicate visit on same url)
 * use concurrentHashMap put to avoid, since the put (insert entry) lock the segment of map
 * and if return null meaning no such key in map previously which means we can process the url
 *
 * 2 - save is a disk I/O where we should put it into a separate thread pool to let it finish by itself
 * 3 - fetch html is a network I/O
 *
 */
class WebCrawlerMultiThread {
    fun crawl(startUrl: String, depth: Int) {
        val visited = ConcurrentHashMap<String, String>()
        val crawlerThreadExecutor = Executors.newFixedThreadPool(THREAD_POOL_MAX_SIZE)
                as ThreadPoolExecutor
        val saveThreadExecutor = Executors.newFixedThreadPool(THREAD_POOL_MAX_SIZE)
                as ThreadPoolExecutor

        val rootCrawlerFuture = crawlerThreadExecutor.submit(InnerCrawler(
                visited,
                crawlerThreadExecutor,
                saveThreadExecutor,
                startUrl,
                1,
                depth
        ))

        rootCrawlerFuture.get()
        crawlerThreadExecutor.shutdown()
        println("====crawler finished===")
        saveThreadExecutor.shutdown()
    }

    class InnerCrawler(
            private val visited: ConcurrentHashMap<String, String>,
            private val executor: ThreadPoolExecutor,
            private val diskWriteExecutor: ThreadPoolExecutor,
            private val url: String,
            private val curDepth: Int,
            private val maxDepth: Int
    ) : Runnable {
        private val nextCrawlers = mutableListOf<Future<*>>()

        override fun run() {
            val html = CrawlerHelper.fetch(url)

            diskWriteExecutor.submit { CrawlerHelper.save(url, html) }

            if (curDepth >= maxDepth) return

            val nextUrls = CrawlerHelper.parse(html)
            nextUrls.forEach { nextUrl ->
                // concurrentHashMap put will lock the map and only current thread can access
                // if return null meaning no same url in the map, thus safe to proceed
                if (visited.put(nextUrl, "") == null) {
                    nextCrawlers.add(executor.submit(InnerCrawler(
                            visited,
                            executor,
                            diskWriteExecutor,
                            nextUrl,
                            curDepth + 1,
                            maxDepth
                    )))
                }
            }
            
            // wait for subthreads to finish
            nextCrawlers.forEach { it.get() }
        }
    }

    companion object {
        private const val THREAD_POOL_MAX_SIZE = 8
    }
}
"""

"""
class KLargestKeyValuePair(private val k: Int) {
    fun map(kvs: List<KeyValuePair>) {
        val keyMaps = mutableMapOf<String, MutableList<Int>>()
        kvs.forEach { pair ->
            keyMaps.computeIfAbsent(pair.key) { _ -> mutableListOf() }.add(pair.value)
        }

        keyMaps.forEach { (key, values) -> MapReduceHelper.writeToFile(key, values) }
    }

    fun reduce(key: String, valueIterator: Iterator<Int>) {
        val kLargest = mergeLargest(valueIterator, k)

        MapReduceHelper.writeToFile(key, kLargest)
    }

    fun reduceWithLargeK(key: String, valueIterator: Iterator<Int>) {
        val curSize = Math.min(MAX_MEM_SIZE, k)
        var iterator = valueIterator
        var remain = k
        var maxNum: Int? = null

        while (remain >= 0) {
            val curLargest = mergeLargest(iterator, curSize, maxNum)
            // write to result file
            MapReduceHelper.writeAppendToFile(key, curLargest)

            maxNum = curLargest.last()
            remain -= curSize
            // reset the iterator and loop in again
            iterator = MapReduceHelper.getIntermediateResult(key)
        }
    }
    
    private fun mergeLargest(valueIterator: Iterator<Int>, size: Int, maxNum: Int? = null): List<Int> {
        val minHeap = PriorityQueue<Int>() { o1, o2 -> o1 - o2 }

        while (minHeap.size < size && valueIterator.hasNext()) {
            val cur = valueIterator.next()
            if (maxNum != null && cur >= maxNum) continue
            minHeap.offer(valueIterator.next())
        }

        while (valueIterator.hasNext()) {
            val curValue = valueIterator.next()

            if (maxNum != null && curValue >= maxNum) continue
            
            if (curValue > minHeap.peek()) {
                minHeap.poll()
                minHeap.offer(curValue)
            }
        }
        
        return minHeap.toList()
    }

    companion object {
        private const val MAX_MEM_SIZE = 1024
    }
}

data class KeyValuePair(val key: String, val value: Int)

object MapReduceHelper {
    fun writeToFile(key: String, values: List<Int>) {}

    fun writeAppendToFile(key: String, values: List<Int>) {}

    fun getIntermediateResult(key: String): Iterator<Int> {
        return listOf<Int>().iterator()
    }
}
"""


4. system design visa payment network
常规的db设计已经如何保证重复的请求不会被处理多次.地里多次讨论这个题目,稍微翻一下就能看见,我就不重复了.
// # https://www.1point3acres.com/bbs/thread-864721-1-1.html

5. 12月面的电面，电面题是那道referal的题，地理面有面经。电面的体验非常好，
感觉他家很在意对题目的分析和与面试官的沟通，先说一下不同解法的trade off，然后再implement。
follow up是如果要算multiple level的refer怎么办，面试官人很nice，一直在引导我找到正确的solution，这步只是聊了想法没有写code

第一轮BQ
第二轮：LazyArray 大致的题目如下
LazyArray a;
a.map(std::function<int>(int) func).indexOf(num)
要考虑到多个map的call，这种情况要把多个function chain起来

求问lazyarray是要indexOf的时候才执行map里的function的意思吗 - yes


第三轮：各种iterators
第四轮：visa payment system，要注意idempotency
第五轮：KV store，可以看一下这个blog，写的很好，这一轮问的很细，算是system design轮


onsite过了以后还有take home assignment，写一个CSV query的parser。这一步之后还要提供三个references。
我是fail在了take home assignment，给我的反馈是code quality不行，没达到production grade，对此我非常无语。
到提供reference这一步也不代表就会过，实在不理解这种方式，不仅耽误面试者的时间，还耽误提供reference的人的时间。

就是给你一个SQL query string，数据都是在给定的几个CSV文件里，返回这个query所要求的数据。要求不光要实现功能，还要重视code quality，test coverage
// # https://www.1point3acres.com/bbs/thread-864697-1-1.html

"""
class NoValueFoundException(message: String) : Exception(message)

// use private constructor to avoid user config the func map
// use static build fun to construct the lazy array
class LazyArray<T> private constructor(
        private val values: List<T>,
        private val prevMapFuncs: List<(input: T) -> T> = emptyList()
) {
    // O(1)
    fun map(mapFunc: (input: T) -> T): LazyArray<T> {
        return LazyArray(values, prevMapFuncs + listOf(mapFunc))
    }

    // O(N * K) - N length of values, K length of map funcs
    // what is multiple value matched?
    fun indexOf(value: T): Int {
        var resultIndex: Int? = null
        var index = 0

        while (index < values.size) {
            var curValue = values[index]

            prevMapFuncs.forEach { func -> curValue = func(curValue) }
            if (curValue == value) {
                resultIndex = index
            }
            index++
        }

        return resultIndex ?: throw NoValueFoundException("No valid value found after mapped.")
    }

    companion object {
        @JvmStatic
        fun <T> build(values: List<T>): LazyArray<T> {
            return LazyArray(values)
        }
    }
}

"""



6. 题目是设计 key value store 能统计 前300s， QPS。https://www.1point3acres.com/bbs/thread-805991-1-1.html
上午面的，面试官是韩裔男，开始感觉沟通挺顺畅，但是等到我implement的时候，他就不断说不理解，思路不对，最后implement完，写了testcase也过了，他也还是说不对，让他给个反例，也没给出来。
当天下午hr就说没通过，反馈是没有implement expected algorithm。反正无语了。
第一次挂电面 orz。反正不知道是沟通有问题还是对方故意为之，我一直都是think loud，在implement之‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌前也是确认过思路的
# https://www.1point3acres.com/bbs/thread-861674-1-1.html

"""
/**
 * use ring buffer to record all the calls in the past 300s
 */
class MockHashMap {
    private val internalMap = mutableMapOf<String, String>()

    private val getLoadRecord = IntArray(RECORD_TIME_ELAPSE_SECOND)
    private val putLoadRecord = IntArray(RECORD_TIME_ELAPSE_SECOND)

    var getPointer: LoadPointer
    var putPointer: LoadPointer

    init {
        val curTime = System.currentTimeMillis() / 1000
        getPointer = LoadPointer(0, curTime)
        putPointer = LoadPointer(0, curTime)
    }

    fun get(key: String): String? {
        logCurrentCall(getPointer, getLoadRecord)
        return internalMap[key]
    }

    fun put(key: String, value: String) {
        logCurrentCall(putPointer, putLoadRecord)
        internalMap[key] = value
    }

    fun measurePutLoad(): Int {
        return getTotalLoad(putPointer, putLoadRecord) / RECORD_TIME_ELAPSE_SECOND
    }

    fun measureGetLoad(): Int {
        return getTotalLoad(getPointer, getLoadRecord) / RECORD_TIME_ELAPSE_SECOND
    }

    private fun getTotalLoad(pointer: LoadPointer, records: IntArray): Int {
        val curTimeS = System.currentTimeMillis() / 1000

        // gap is how many slots we need move backforward from cur slot to count the total load
        var gaps = RECORD_TIME_ELAPSE_SECOND - (curTimeS - pointer.timestampS)
        if (gaps <= 0) {
            return 0
        }

        var count = 0
        var mover = pointer.index
        while (gaps > 0) {
            count += records[mover]
            mover = getNextMoveIndexBackward(mover - 1, RECORD_TIME_ELAPSE_SECOND)
            gaps--
        }

        return count
    }

    private fun getNextMoveIndexBackward(nextIndex: Int, size: Int): Int {
        return if (nextIndex < 0) {
            size + nextIndex
        } else {
            nextIndex
        }
    }

    private fun logCurrentCall(pointer: LoadPointer, records: IntArray) {
        val curTimeS = System.currentTimeMillis() / 1000

        if (curTimeS == pointer.timestampS) {
            records[pointer.index] ++
            return
        }

        // reset the gap between to 0
        val gap = curTimeS - pointer.timestampS
        val newIndex = ((pointer.index + gap) % RECORD_TIME_ELAPSE_SECOND).toInt()
        if (gap >= RECORD_TIME_ELAPSE_SECOND) {
            resetRecordSlot(0, records.size - 1, records)
        } else {
            // make sure the start index is within range of records
            resetRecordSlot((pointer.index + 1) % RECORD_TIME_ELAPSE_SECOND, newIndex, records)
        }

        // update
        records[newIndex] = 1
        pointer.index = newIndex
        pointer.timestampS = curTimeS
    }

    private fun resetRecordSlot(startIndex: Int, endIndex: Int, record: IntArray) {
        val gap = endIndex - startIndex + 1
        var moveCount = if (gap >= 0) gap else RECORD_TIME_ELAPSE_SECOND + gap
        var mover = startIndex

        while (moveCount > 0) {
            record[mover] = 0
            mover = (mover + 1) % RECORD_TIME_ELAPSE_SECOND

            moveCount--
        }
    }

    companion object {
        private const val RECORD_TIME_ELAPSE_SECOND = 10
    }
}

data class LoadPointer(
        var index: Int,
        var timestampS: Long
)
"""

7. Round 0 - Hiring Manager Call. 聊一些项目，看看是否match team。
Onsite：
Round 1 - Design&Coding.
设计一个单机的 kv store 数据库.
1. API: 同步put/get, key ~100 字节, value ~1000 bytes
2. Keys and Values can be saved in the memory.
3. Be able to recover from machine reboot.
Round 2 - Design
设计一个分布式文件系统,可以用zk,key-value store,但是不能用S3等现成的产品。
1. putFile(path, data)/getFile(path, offset, length)/deleteFile(path)
2. listDirectory(path)
Round 3 - Behavior. 扯之前的项目，从头到尾叙述整个项目，问的很细。问一些人/项目的问题。
Round 4 - Coding. 写一个me‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌tric系统，统计前5分钟不同API的avg QPS。写完之后扩展到任意时间范围
Round 5 - Coding. 实现懒惰Array，提供map(lambda)/indexOf(value).
# https://www.1point3acres.com/bbs/thread-856819-1-1.html

8. system design:
Design a VISA payment system for an ecommerce website.
给了一个isser bank 和一个merchant bank. 要求design 这个 payment system本身，和payment system 与 isser bank 和 merchant bank 对接。
# https://www.1point3acres.com/bbs/thread-854235-1-1.html

9. 1月初店面：三大经典面试题其中之一
1月初HM：聊project

system design: web crawler multithreaded
coding1: 非面经题 一个复杂的bfs 题不难 但是比较难写。一个metric代表location, 有起点和终点，
有四种不同的出行方式，每种出行方式经过一个点有不同的time和cost，求起点到终点的最短时间（时间相同则最少cost）中途不能换交通方式。
 Followup：中途可以换交通方式
coding2: 非面经题 具体的忘了 大概是有数据库两个replica，master and follower, 每一个replica有一个lock，
连续两次failure request lock 就会锁起来然后 reject request, lock在锁起来的时候连续两次reject request后锁就会打开。
Request会先去master在master reject/fail request后会去follower，要实现这个逻辑并且跑通。
BQ
2月初 take home - CSV query和reference call
# https://www.1point3acres.com/bbs/thread-852792-1-1.html

"""
//非面经题 一个复杂的bfs 题不难 但是比较难写。一个metric代表location, 有起点和终点，
//有四种不同的出行方式，每种出行方式经过一个点有不同的time和cost，求起点到终点的最短时间（时间相同则最少cost）中途不能换交通方式。
//Followup：中途可以换交通方式

class UnableToReachDestinationError(message: String) : Exception(message)

// use dijkstra's algorithm
// maintain a heap in which is the node with min travel time/cost
// pop and check neighbors, add neighbors in
// util find destination
// time complexity:
// map size M * N
// dijkstra complexity is E(edge) + VlogV(nodes)
// thus here is (2MN + M + N + MNlogMN)
class LowestCostTravel {
    fun findLowestCostTravelMethod(
            start: Point,
            end: Point,
            travel: Array<Array<Char>>,
            methodTime: Array<Int>,
            methodCost: Array<Int>
    ): Char {
        if (travel.isEmpty() || (start.x == end.x && start.y == end.y)) {
            return '-'
        }
        val height = travel.size
        val width = travel[0].size

        val visitedMap = Array<Array<Boolean>>(height) { Array(width) { false } }
        val minHeap = PriorityQueue<CostPoint>() { o1, o2 ->
            if (o1.cost.time == o2.cost.time) {
                o1.cost.cost - o2.cost.cost
            } else {
                o1.cost.time - o2.cost.time
            }
        }

        // init
        visitedMap[start.x][start.y] = true
        directions.forEach initStart@{ dir ->
            val nextX = dir.x + start.x
            val nextY = dir.y + start.y
            if (
                    checkPointToSkip(nextX, nextY, visitedMap, height, width)
                    || travel[nextX][nextY] == 'x'
            ) {
                return@initStart
            }

            minHeap.offer(
                    CostPoint(
                            Point(nextX, nextY),
                            travel[nextX][nextY],
                            TravelCost(0, 0)
                    )
            )
        }

        // travel
        while (minHeap.isNotEmpty()) {
            val curCostPoint = minHeap.poll()
            val curPoint = curCostPoint.point
            val curMethod = curCostPoint.method

            // mark visited here
            visitedMap[curPoint.x][curPoint.y] = true

            if (curPoint.x == end.x && curPoint.y == end.y) {
                return curMethod
            }

            // check neighbors
            directions.forEach directionLoop@{ direction ->
                val nextX = curPoint.x + direction.x
                val nextY = curPoint.y + direction.y
                if (
                        checkPointToSkip(nextX, nextY, visitedMap, height, width)
                        || (travel[nextX][nextY] != 'D' && travel[nextX][nextY] != curMethod) // need to go to same method
                ) {
                    return@directionLoop
                }

                minHeap.offer(
                        CostPoint(
                                Point(nextX, nextY),
                                curMethod,
                                TravelCost(
                                        curCostPoint.cost.time + methodTime[curMethod - '1'],
                                        curCostPoint.cost.cost + methodCost[curMethod - '1']
                                )
                        )
                )
            }
        }

        throw UnableToReachDestinationError("unable to reach destination")
    }

    private fun checkPointToSkip(
            x: Int,
            y: Int,
            visitedMap: Array<Array<Boolean>>,
            height: Int,
            width: Int
    ): Boolean {
        return x < 0 || x >= height || y < 0 || y > width || visitedMap[x][y]

    }

    companion object {
        private val directions = arrayOf(
                Point(1, 0),
                Point(-1, 0),
                Point(0, 1),
                Point(0, -1)
        )
    }
}

data class TravelCost(val time: Int, val cost: Int)

data class CostPoint(val point: Point, val method: Char, val cost: TravelCost)

data class Point(val x: Int, val y: Int)

"""


10. 电面：非面经题，不难，类似糁巴斯
1.  BQ
2. 非面经，简单DP。给定整数数组，不能取相邻的数，使能取到的数之和最大。 [9,1,4] -> 13
3. llvm - 
# https://leetcode.com/discuss/interview-question/1227652/data&#8205;&#8205;&#8204;&#8205;&#8204;&#8205;&#8204;&#8204;&#8204;&#8204;&#8204;&#8204;&#8204;&#8205;&#8205;&#8204;&#8204;bricks-special-language-llvm-round-coding-debugging-testing
4. visa payment
5. multithreading web crawler   伊尔斯尔

股票交易系统，第三方交易平台提供现成的接口有
1. POST 下单 -》 返回‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌给你订单号
2. GET 按订单号查询
3. DELETE 按订单号删除
我需要设计一个系统在第三方交易平台上实现买和卖股票两个功能，难点在于买卖都要支持”交易自动终止时间“功能。 - 接口要支持一个时间的变量，时间到了，交易需要取消
# https://www.1point3acres.com/bbs/thread-849020-1-1.html

"""
class ShuffleArray {
    private int[] original;
    private int[] array;
    private Random random = new Random();
    public Solution(int[] nums) {
        this.array = nums;
        this.original = nums.clone();
    }
    
    private List<Integer> getArrayCopy() {
        List<Integer> asList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            asList.add(array[i]);
        }
        return asList;
    }
    
    public int[] reset() {
        array = original;
        original = original.clone();
        return array;
    }
    
    public int[] shuffle() {
        List<Integer> helper = getArrayCopy();
        
        for (int i = 0; i < array.length; i++) {
            int randomIndex = random.nextInt(helper.size());
            array[i] = helper.get(randomIndex);
            helper.remove(randomIndex);
        }
        
        return array;
    }
}

"""

"""
rob house??
// DP new array each slot save the max sum in current size of the array.
// next position's element in input just need to get the max sum by
// MAX(cur_element + max_sum[i - 2], max_sum[i - 1])
// time complexity O(N)
class LargestSum {
    fun getLargestSum(input: List<Int>): Int {
        if (input.isEmpty()) {
            return 0
        } else if (input.size == 1) {
            return input[0]
        }

        val sums = Array<Int>(input.size) { 0 }
        sums[0] = input[0]
        sums[1] = Math.max(sums[0], input[1])

        var index = 2
        while (index < sums.size) {
            sums[index] = Math.max(sums[index - 1], input[index] + sums[index - 2])

            index++
        }

        return sums.last()
    }
}
"""


11. 
在地里没见过，po 出来帮助后面的同学。其实题不难。但是自己脑子浆糊了
Assume we have some static, globally available reference string
// ref_string = ['a', 'b', 'c', '1', '2', '3', '4', 'a', 'b', 'c', 'd', '!'] == "abc1234abcd!"
//    (index) =   0    1    2    3    4    5    6    7    8    9    10   11
// Using the reference string, we want to compress a source string
// src_string = ['a', 'b', 'c', 'd', '1', '2', '3'] == "abcd123"
//    (index) =   0    1    2    3    4    5    6
// A cover represents a compression of src_string relative to ref_string and is
// comprised of (inclusive, exlcusive) indicies-pairs called "blocks". For example,
// block1 = (7, 11) => "abcd"
// For example, one valid cover for src_string:
// cov1 = [(7, 11), (3, 6)] => ["abcd", "123"]
// Another valid cover for src_string:
// cov2 = [(7, 10), (10, 11), (3, 6)] => ["abc", "d", "123"]
// Implement delete(cover, index)
// Given a valid cover and index of S, return a valid cover for S[:index] + S[index+1:]
// Array[Array[Int]] delete(Array[Array[Int]] cover, Int index)
// cov1 = [(7, 11), (3, 6)]
// abc123
// abcd13
// delete(cov1, 3) = [(7,10), (3,6)}]
// delete(cov1, 5)
// delete(cov1, 0)
// delete(cover1, 3) -> (7,10), (3,6)
/*
*  Follow- up,  delete(cover, index, ref_str); but you need to return maxim cover
*/
// A "maximal" cover is one in which concatenating any consecutive pair of blocks
// yields a corresponding substring that is not in the reference string.
// cov1 is maximal since ("abcd" + "123") or "abcd123" is not in ref_string
//cov1 = [(7, 11), (3, 6)] => ["a‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌bcd", "123"]
// cov2 is NOT maximal since ("abc" + "d") or "abcd" is in ref_string
// cov2 = [(7, 10), (10, 11), (3, 6)] => ["abc", "d", "123"]
// delete(cov1, 3) = [(7,10), (3,6)}] = "abc""123" = (0,6)

# https://www.1point3acres.com/bbs/thread-847914-1-1.html

"""
class StringCoverDelete {
    fun delete(covers: List<Array<Int>>, index: Int): List<List<Int>> {
        val refCharIndexMap = buildCharIndexMap(REF_STRING)

        val actualStringBuilder = StringBuilder()
        covers.forEach { cover ->
            actualStringBuilder.append(REF_STRING.substring(cover[0], cover[1]))
        }
        actualStringBuilder.delete(index, index + 1)
        val actualString = actualStringBuilder.toString()

//        return findNormalCover(actualString, refCharIndexMap)
        return findMaxCover(actualString, refCharIndexMap)
    }

    // greedy move forward, if match, expand the cover. otherwise just add into result
    private fun findNormalCover(
            actualString: String,
            refCharIndexMap: Map<Char, MutableList<Int>>
    ): List<List<Int>> {
        val result = mutableListOf<MutableList<Int>>()

        actualString.toCharArray().forEach { char ->
            val charIndices = refCharIndexMap[char]!!
            val firstIndex = charIndices.first()

            if (result.isEmpty() || result.last()[1] != firstIndex) {
                result.add(mutableListOf(charIndices.first(), charIndices.first() + 1))
            } else {
                result.last()[1]++
            }

            charIndices.remove(charIndices.first())
        }

        return result
    }

    // bfs find all solutions, shortest one is the one cover all
    private fun findMaxCover(
            actualString: String,
            refCharIndexMap: Map<Char, MutableList<Int>>
    ): List<List<Int>> {
        val actualCharArray = actualString.toCharArray()
        val queue: Queue<CoverList> = LinkedList()

        refCharIndexMap[actualCharArray.first()]!!.forEach { index ->
            queue.offer(CoverList(mutableListOf(mutableListOf(index, index + 1)), mutableSetOf()))
        }

        var mover = 1
        while (mover < actualCharArray.size) {
            val curIndices = refCharIndexMap[actualCharArray[mover]]!!
            var levelSize = queue.size
            while (levelSize > 0) {
                val curCoverList = queue.poll()

                curIndices.forEach indexLoop@{ index ->
                    if (curCoverList.visited.contains(index)) {
                        return@indexLoop
                    }

                    val newCoverList = ArrayList(curCoverList.list)
                    val newVisited = HashSet(curCoverList.visited)
                    if (index == newCoverList.last()[1]) {
                        newCoverList.last()[1]++
                    } else {
                        newCoverList.add(mutableListOf(index, index + 1))
                    }
                    newVisited.add(index)

                    queue.offer(CoverList(newCoverList, newVisited))
                }

                levelSize--
            }

            mover++
        }

        return queue.toList().map { it.list }.minWithOrNull(Comparator.comparingInt { it.size }) ?: emptyList()
    }

    private fun buildCharIndexMap(string: String): Map<Char, MutableList<Int>> {
        val map = mutableMapOf<Char, MutableList<Int>>()

        string.toCharArray().forEachIndexed { index, char ->
            map.computeIfAbsent(char) { _ -> mutableListOf() }.add(index)
        }

        return map
    }

    companion object {
        private const val REF_STRING = \"abc1234abcab\"
    }
}

data class CoverList(val list: MutableList<MutableList<Int>>, val visited: MutableSet<Int>)

"""


12. 
题目与https://www.1point3acres.com/bbs/thread-844563-1-1.html完全一致
几点注意:
1. topk 需要按照map和reduce分别写. 然后是考虑k大和n大的情况
2. 懒数组需要分别实现那几个interface. 尤其需要考虑连环的情况
3. kv就是考察wal和加锁的情况. 这里有一个例子楼主觉得不错.供大家参考.https://martinfowler.com/article ... ed-systems/wal.html
# https://www.1point3acres.com/bbs/thread-845769-1-1.html

13. 

老题。上来问了这个帖子(https://www.1point3acres.com/bbs/thread-812809-1-1.html)
里面的第三题(KV store). 大概说了几种方法。然后面试官说这个题你好像见过，我说那你换吧。。。
然后出了一个ip allow/deny的题。好像面经里面提到过,我一时找不到了。就是给一个list of rules, 判断最后是allow还是deny.
不用处理conflict,看到第一个是什么就返回。boolean isAllowed(String ip, String[][] rules); 没有技巧，知道什么是子网掩码的概念就行，硬写。
[
["192.168.0.1/32", "ALLOW"],
["192.168.0.4/30", "DENY"],
...
]

# https://www.1point3acres.com/bbs/thread-844718-1-1.html
"""
// int is 32 bits but range is -2^31 ~ 2^31 - 1 
// where the ip is 2^32 thus use Long to represent number of ip
class IpCIDRMask {
    fun isAllowed(ip: String, rules: List<Array<String>>): Boolean {
        val ipNum = convertToNum(ip)

        var allow = false
        rules.forEach { rule ->
            val mask = convertRuleToMask(rule[0])
            if (ipNum >= mask.ipNum && ipNum <= mask.ipNum + mask.range) {
                allow = parseRule(rule[1])
                // return@forEach
            }
        }

        return allow
    }

    private fun parseRule(rule: String): Boolean {
        return when (rule) {
            "ALLOW" -> true
            "DENY" -> false
            else -> false
        }
    }

    private fun convertToNum(ip: String): Long {
        val numStrs = ip.split(".")
        var number = 0L
        numStrs.forEach { numStr ->
            number = number * 256 + numStr.toLong()
        }

        return number
    }

    private fun convertRuleToMask(mask: String): Mask {
        val numStrs = mask.split("/")[0]
        val maskStr = mask.split("/")[1]

        val num = convertToNum(numStrs)
        val range = 2.0.pow((32 - maskStr.toInt()).toDouble()) - 1

        return Mask(num, range.toLong())
    }
}

data class Mask(val ipNum: Long, val range: Long)

"""


14. 
1. design: k-v store.  Followup: WAL, snapshot
2. design: visa payment system, 看到了面筋，但是没怎么准备，估计是挂这上面了
3. coding: lazy array operations, how to test the implementation
4. coding: top k value per key, how to deal with hotspot
5. BQ
# https://www.1point3acres.com/bbs/thread-844563-1-1.html

15. 
1. web crawler 问的比较细，各种disk io，network io优化
2. 设计lazyarray 之前面经里有
3. 设计MockHashMap, 面经题，也是问的很细，很多followup
4. hm
5. visa payment network
# https://www.1point3acres.com/bbs/thread-843407-1-1.html

16. 
给两个inputstream，已经排好序，合并相同的key，并且把values进行归并求和。具体题目如下，一般是按merge sort的方式写，这样是linear time和O（1）空间，
但麻烦的重复的元素和尾部处理，调试了半天也没通过，希望大家指点一下
"""
There are two large files containing SORTED key value pairs, where keys are strings and values are integers. e.g.
File 1:
aaa: 1
bbb: 5
bbb: 3
ccc: 2
File 2:
bbb: 9
ccc: 10
ddd: 11
We want to merge the two files together to produce an output file where keys are still sorted. Consecutive pairs with the same key in the output are merged, by summing up their values. e.g. merging the files above produces the output:
aaa: 1
bbb: 17
ccc: 12
ddd: 11
"""
// # API to read from the input file. Example usage:
// #
// #   while input_stream.is_valid():
// #       key, val = input_stream.read()
// #       ......
// #       input_stream.next()

// """
// class InputStream:
//     # For testing, InputStream can be constructed from a list.
//     def __init__(self, data):
//         self._data = data
//         self._current = 0
//     # Checks whether the stream has data at the current position.
//     # Returns false if the stream is already ended.
//     def is_valid(self):
//         return self._current < len(self._data)
//     # Gets the current pair.
//     # Returns None if the stream is already ended.
//     def read(self):
//         if self.is_valid():
//             return self._data[self._current]
//         return None
//     # Advances to the next item in the stream.
//     def next(self):
//         if self.is_valid():
//             self._current += 1
// # API to write to the output file.
// # During testing, data written can be accessed via the data() method.
// class OutputStream:
//     def __init__(self):
//         self._data = []
//     # Writes pair to output file.
//     def write(self, data):
//         self._data.append(data)
//     # For testing, OutputStream data is save‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌d in memory.
//     def data(self):
//         return self._data
// import collections
// # Implement this:
// def merge_input_files(input_1: InputStream, input_2: InputStream, output: OutputStream):
//    pass
// #### test ########
// data1 = (("aaa", 1), ("bbb", 5), ("bbb", 3), ("ccc", 2))
// data2 = (("bbb", 9), ("ccc", 10), ("ddd", 11))
// input1 = InputStream(data1)
// input2 = InputStream(data2)
// output = OutputStream()
// merge_input_files(input1, input2, output)
// print("-----")
// print(output._data)
// """
重复元素你可以不马上塞到output里。用一个key value pair暂时存着，直到input1和input2里挑出来的新的key value pair的key和暂时存的不同
才把之前的pair塞进output里。如果是重复key，那就直接更新暂存的key value pair。
尾部没啥特别需要处理的吧，就是保证一个input结束之后另一个input还能继续。

# https://www.1point3acres.com/bbs/thread-842970-1-1.html

"""
// merge sort
// use tempPair to store if there are duplicate keys in the files
// and only log tempPair into output
// when both pointer is ahead of tempPair key, log tempPair into output
// and re-assign tempPair to smaller key (or sum together if value equal)
// time complexity: O(N + M)
// space complexity: O(1)

data class ReaderPair(val curPair: Pair<String, Int>, val reader: FileReader)

class MergeSortedFiles {
    fun mergeK(readers: List<FileReader>, output: OutputStream) {
        var tempPair: Pair<String, Int>? = null

        val minHeap = PriorityQueue<ReaderPair> { o1, o2 -> o1.curPair.first.compareTo(o2.curPair.first) }
        readers.forEach { reader ->
            if (reader.hasNext()) minHeap.offer(ReaderPair(reader.next(), reader))
        }

        while (minHeap.isNotEmpty()) {
            val curReaderPair = minHeap.poll()
            val curPair = curReaderPair.curPair

            tempPair = if (tempPair == null) {
                Pair(curPair.first, curPair.second)
            } else if (tempPair.first == curPair.first) {
                Pair(tempPair.first, tempPair.second + curPair.second)
            } else {
                output.write(tempPair)
                Pair(curPair.first, curPair.second)
            }

            if (curReaderPair.reader.hasNext()) {
                minHeap.offer(ReaderPair(curReaderPair.reader.next(), curReaderPair.reader))
            }
        }

        if (tempPair != null) {
            output.write(tempPair)
        }
    }


    fun merge(file1: FileReader, file2: FileReader, output: OutputStream) {
        var tempPair: Pair<String, Int>? = null

        var file1Pair = getNextFileRecord(file1)
        var file2Pair = getNextFileRecord(file2)
        while (file1Pair != null || file2Pair != null) {
            if (file1Pair == null) {
                // log and assign new temp pair or stack the value from file2Pair 
                tempPair = assignTempPair(tempPair, file2Pair!!, output)
                file2Pair = getNextFileRecord(file2)
                continue
            } else if (file2Pair == null) {
                // log and assign new temp pair or stack the value from file1Pair 
                tempPair = assignTempPair(tempPair, file1Pair!!, output)
                file1Pair = getNextFileRecord(file1)
                continue
            }

            if (file1Pair.first > file2Pair.first) {
                tempPair = assignTempPair(tempPair, file2Pair, output)
                file2Pair = getNextFileRecord(file2)
            } else if (file1Pair.first < file2Pair.first) {
                tempPair = assignTempPair(tempPair, file1Pair, output)
                file1Pair = getNextFileRecord(file1)
            } else {
                tempPair = if (tempPair == null) {
                    Pair(file1Pair.first, file1Pair.second + file2Pair.second)
                } else if (file1Pair.first == tempPair.first) {
                    // if all equals
                    Pair(file1Pair.first, tempPair.second + file1Pair.second + file2Pair.second)
                } else {
                    // if tempPair is smaller
                    output.write(tempPair)
                    Pair(file1Pair.first, file1Pair.second + file2Pair.second)
                }

                file1Pair = getNextFileRecord(file1)
                file2Pair = getNextFileRecord(file2)
            }
        }

        if (tempPair != null) {
            output.write(tempPair)
        }
    }

    private fun assignTempPair(
            curTempPair: Pair<String, Int>?,
            filePair: Pair<String, Int>,
            output: OutputStream
    ): Pair<String, Int> {
        // if temp not null and equals, stack the value. no need to log. just return
        if (curTempPair != null && filePair.first == curTempPair.first) {
            return Pair(curTempPair.first, curTempPair.second + filePair.second)
        }
        // if not empty, log and value and assign new temp
        if (curTempPair != null) {
            output.write(curTempPair)
        }
        return Pair(filePair.first, filePair.second)
    }

    private fun getNextFileRecord(fileReader: FileReader): Pair<String, Int>? {
        return if (fileReader.hasNext()) fileReader.next() else null
    }
}

class FileReader(fileValue: List<Pair<String, Int>>) {
    private val iterator = fileValue.iterator()

    fun hasNext(): Boolean {
        return iterator.hasNext()
    }

    fun next(): Pair<String, Int> {
        return iterator.next()
    }
}

class OutputStream {
    private val data = mutableListOf<Pair<String, Int>>()

    fun write(pair: Pair<String, Int>) {
        data.add(pair)
    }

    fun data(): List<Pair<String, Int>> {
        return data
    }
}

"""


17. 
店面：MockHashMap follow up问了内存不足时如何压缩
VO 第一轮：Visa payment system
VO 第二轮：llvm
VO 第三轮：SnapshotSet
VO 第四轮：Webcrawler, 可以写伪代码，注意concurrency.
VO 第五轮：BQ，聊项目，注意学会夸数据🧱‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌的人都很smart

MockerHashmap地里有原题
SnapshotSet就是除了set的基本api，还需要实现一个返回Snapshot的功能，可以用来遍及当前所有元素

# https://www.1point3acres.com/bbs/thread-842603-1-1.html

18. 
第一轮 新题 实现lazyArray 就是实现java里面的array.map(a->XXX).indexOf(); 要求map操作推迟到最后才做。所以叫lazy
第二轮 设计durable kv store 一个老年烙印 假装很耐撕 其实一看就不想给过。我说要类似kafka可以replay msg events, 他说你不能用kafka, 要自己实现。
我说只是想解释下这个原理，不然design讲啥？
接着开始写码。结果一边说这轮是low level design要写code，写一半又让我别写了，说太多代码。
这玩意redis都有很多成熟的实现了 你觉得你们很懂很高级吗？
第三轮 维萨支付系统那个出烂了的题。 again, 砖厂好像很懂payment？
1第五轮 BQ
没碰到LLVM  为准备这破题花了好长时间看编译语言 最后没有考。。。

LLVM - https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=709588&ctid=232643

// # https://www.1point3acres.com/bbs/thread-832280-1-1.html


19. 
给定一系列规则：
[("ALLOW","1.2.3.4/32"),
("DENY","5.6.7.8/30"),
...
]
判断一个IP最终是ALLOW还是DENY，返回True或False。
一开始clarification做得还不错，问了整个list有没有overlap，ALLOW和DENY有没有overlap，如果有以谁为准，是不是只有ALLOW和DENY两个种类的规则，有没有非法输入等。
最终把问题转化为判断一个IP是否在CIDR中。这部分很顺。
但是开始写这个判断的函数时，对方说不能用任何库。十进制转换成二进制也得自己写。于是我一个一个手写，用除以2的方法手写出来了。
然后求出CIDR的起始和截止IP，最后判断IP是否在其中（其实后来想了想，只用判断IP和CIDR的前n位是否相同就可以确定了），
然后结合整个的规则列表判断最终结果。这部分耗时很长，一个接一个的helper function，但好在思路没有卡壳，一直在慢慢往前走。
最后做出来了，但是中途有很多低级错误，如：十进制转换二进制的函数，忘记写return了。。。再比如，固定的位数（/30）为str类型，没及时转化成int，又报错，
后来改的时候糊涂了，用了个len()，还是小哥给改成int()。

这个直接把字符串转换为十进制数用右移就好了，还去转换成二进制表达式有点麻烦
对的，这个我之后也看过leetcode，确实是你说的这种解法。我是属于有deadline，把主要章节刷完了就直接上的。但是有个专题叫bit manipulation，我以为属于很偏的考点，就直接略过了。

// # https://www.1point3acres.com/bbs/thread-828501-1-1.html


20. 
面的题是key-value store， 有 get和put两种method要实现。然后要返回5min之内call了多少次get和put。
// # https://www.1point3acres.com/bbs/thread-880376-1-1.html


21.
// 电面我选择了backend，题目是customer revenue，地里的高频题

// 第一轮coding，mock hashmap，之前地里出现过
// 第二轮coding，
// lazy array，要求实现 array.map(...).indexOf(...)，其中map传进去一个function，
// indexOf返回运行了所有function之后传入值的index。要求map的操作最后再做，所以叫lazy array。For example:
// arr = LazyArray([10, 20, 30, 40, 50])
// arr.map(lambda x:x*2).indexOf(40)  ----> 1
// arr.map(lambda x:x*2).map(lambda x:x*3).indexOf(240) ----> 3 注意这里重新开了一个chain，上一行的map就不计算在内了
// 第三轮HM，就是普通的BQ，比如有没有跟队员有conflict，最大的优点和缺点，what are you looking for in your next role之类的
// 第四轮system design，其实是javascript coding‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌，要求写一个autocomplete widget的onInputChange handler，不需要考虑view，只需要update state就可以了。考了各种可能出现的error。
// # https://www.1point3acres.com/bbs/thread-879121-1-1.html

22.
// 第一问 类似与 器零散
// 但是给的输入是 k = 2：
// [a, 1]
// [a, 2]
// [a, 3]
// [a, 4]
// [a, 5]
// [b,7]
// [b, 8]
// 要求输出是
// [a, 5], [a4]
// [b, 7], [b, 8]
// 第二问是给了两个map reduc‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌e的api，用map reduce求解

# https://www.1point3acres.com/bbs/thread-878844-1-1.html

23.
不是面筋题 给一个matrix 每个格子有权值，找出权值最短路径，左上到右下，dp解决，应该是离口原题不记得题号了，后面闲聊。
# https://www.1point3acres.com/bbs/thread-876441-1-1.html

24.
题就是 product of  sparse matrix
怎么表达sparse matrix我用map< row number,   map< column number,  no-zero value>>
他给了一个idea是用3个数组表示；val[]， col[]， row[]
花了很多时间交流和理解怎么计算一个no zero element的row
最后为了节省时间他让只让我写了pseudo code ，只写 product计算部分 默认 sparse matrix 已经转化成了 三个数组。

# https://www.1point3acres.com/bbs/thread-825770-1-1.html
"""
// use hashmap to store for easier find corresponding slot
// map <row, <map<col, val>>> for first matrix
// or map <col, <map<row, val>>> for second matrix
public class Solution {
    public int[][] multiply(int[][] A, int[][] B) {
        // Search in A find who is not 0, and go into B find in Kth row who is not 0.then C[i][j] += A[i][k] * B[k][j]
        int[][] C = new int[A.length][B[0].length];
        
        Map<Integer, Map<Integer, Integer>> compressA = compressMatrix(A, true);
        Map<Integer, Map<Integer, Integer>> compressB = compressMatrix(B, false);
        
        compressA.keySet().forEach( 
            row -> {
                compressB.keySet().forEach(
                    col -> {
                        Map<Integer, Integer> rowVals = compressA.get(row);
                        Map<Integer, Integer> colVals = compressB.get(col);

                        rowVals.keySet().forEach(
                            colA -> {
                                if (!colVals.containsKey(colA)) {
                                    return;
                                }
                                Integer valA = rowVals.get(colA);
                                Integer valB = colVals.get(colA);
                                C[row][col] += valA * valB;    
                            }
                        );       
                    } 
                );    
            }       
        );
        
        return C;
    }
    
    private Map<Integer, Map<Integer, Integer>> compressMatrix(int[][] matrix, boolean firstMatrix) {
        // map <row, <map<col, val>>> for first matrix
        // or map <col, <map<row, val>>> for second matrix
        Map<Integer, Map<Integer, Integer>> compressed = new HashMap<>();
        
        for (int row = 0; row < matrix.length; row++) {
            for (int col = 0; col < matrix[0].length; col++) {
                if (matrix[row][col] != 0) {
                    if (firstMatrix) {
                        compressed.computeIfAbsent(
                            row, key -> new HashMap<>()
                        ).put(col, matrix[row][col]);
                    } else {
                        compressed.computeIfAbsent(
                            col, key -> new HashMap<>()
                        ).put(row, matrix[row][col]);
                    }
                }
            }
        }
        
        return compressed;
    }
}
"""
// OR use the 3 arrays to store the matrix.
// the key to easier locate the non-zero elements in corresponding column in 2nd matrix by
// the row of first matrix
// time complexity in worst case is always O(n*k*m) and space O(n*k + k*m)
"""
public class Solution {
    class CompressedMatrix {
        List<Integer> value;
        List<Integer> column;
        List<Integer> row;
        
        public CompressedMatrix(List<Integer> val, List<Integer> col, List<Integer> row) {
            this.value = val;
            this.column = col;
            this.row = row;
        }
    };
    
    private CompressedMatrix compressMatrixOnRow(int[][] matrix) {
        List<Integer> val = new ArrayList<>();
        List<Integer> column = new ArrayList<>();
        List<Integer> row = new ArrayList<>();
        
        if (matrix.length == 0) {
            return null;
        }
        
        row.add(0);
        for (int rowIndex = 0; rowIndex < matrix.length; rowIndex++) {
            for (int colIndex = 0; colIndex < matrix[0].length; colIndex++) {
                if (matrix[rowIndex][colIndex] != 0) {
                    val.add(matrix[rowIndex][colIndex]);
                    column.add(colIndex);
                }
            }
            row.add(val.size());
        }
        
        return new CompressedMatrix(val, column, row);
    }
    
    private CompressedMatrix compressMatrixOnCol(int[][] matrix) {
        List<Integer> val = new ArrayList<>();
        List<Integer> column = new ArrayList<>();
        List<Integer> row = new ArrayList<>();
        
        if (matrix.length == 0) {
            return null;
        }
        
        column.add(0);
        for (int colIndex = 0; colIndex < matrix[0].length; colIndex++) {
            for (int rowIndex = 0; rowIndex < matrix.length; rowIndex++) {
                if (matrix[rowIndex][colIndex] != 0) {
                    val.add(matrix[rowIndex][colIndex]);
                    row.add(rowIndex);
                }
            }
            column.add(val.size());
        }
        
        return new CompressedMatrix(val, column, row);
    }
    
    public int[][] multiply(int[][] A, int[][] B) {
        // Search in A find who is not 0, and go into B find in Kth row who is not 0.then C[i][j] += A[i][k] * B[k][j]
        int[][] C = new int[A.length][B[0].length];
        
        CompressedMatrix compA = compressMatrixOnRow(A);
        CompressedMatrix compB = compressMatrixOnCol(B);
        
        for (int row = 0; row < C.length; row++) {
            for (int col = 0; col < C[0].length; col++) {
                int rowStart = compA.row.get(row);
                int rowEnd = compA.row.get(row + 1);
                
                int colStart = compB.column.get(col);
                int colEnd = compB.column.get(col + 1);
                
                // find index matched in 2 sorted array
                // value must match on column(rowIndex)/row(colIndex) array
                while (rowStart < rowEnd && colStart < colEnd) {
                    if (compA.column.get(rowStart) < compB.row.get(colStart)) {
                        rowStart++;
                    } else if (compA.column.get(rowStart) > compB.row.get(colStart)) {
                        colStart++;
                    } else {
                        C[row][col] += compA.value.get(rowStart) * compB.value.get(colStart);
                        rowStart++;
                        colStart++;
                    }
                }
            }
        }
        
        return C;
    }
"""


25.
coding：用LLVM的IR实现整数除法，用naive的减法就可以了。followup的问题是如何处理负数，没要求写出代码。
system design：设计一个durable key/value store。要求写主要代码，不要求分布式。


26.
题目大概就是shortest path in a 2d matrix.
现在看来就是以下两道Leetcode套上一点马甲：力扣 二八刘
力扣 三一起

# https://www.1point3acres.com/bbs/thread-824812-1-1.html

27.
1. bq 烙印manager 很harsh 给我的感觉就是要搞我 大概率也确实把我搞了。特别喜欢challenge 我。
离谱的是最后问他喜欢和不喜欢什么，他说最喜欢的地方是公司把提高招人标准当公司priority。
2. low level system design/coding 爬虫 人很好，交流挺好
3. high level system design 国人大哥 大哥很忙碌的样子 但是人很友善也很engaged。感觉没什么特别要design的，
讨论了很久各种常见和比较贴近工作里会遇到的各种tradeoff。
4. coding 面这轮时候感觉前三轮还不错 以为是llvm。然而并不是，是一题简单算法。有统计和ML的背景的话，会很容易。小哥人很nice。
但是我们出现了一些miscommunication 耽误了很多时间。导致后面我的状态崩了，有bug也没修出来。这轮fail
5. coding 这轮也不是llvm。。。是一道新题。有点像是用spark lazy execution的理念实现些interface。不难。

# https://www.1point3acres.com/bbs/thread-822481-1-1.html

28.
【對話筐對齊】是 LC 流失巴 这道题么

给定每一行的单字跟页宽, 输出要跟對話框一樣輸出
輸入是 [int, string]
Int 只有 1 2 代表兩個人名
1的名字打的內容要靠左
2的名字打的內容要靠右
像通訊 軟體那樣輸出

# https://www.1point3acres.com/bbs/thread-820173-1-1.html

29.
店面劈頭就是design question：設計一個可以多人同步編輯的 music play list
基本上面試官互動也不多也不引導，完全不知道他想要的答案是什麼
按照自己的設計進行討論然後隔天就被拒了
該講的東西也都講了，api method payload, web sock‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌et broadcast subscribe

# https://www.1point3acres.com/bbs/thread-818999-1-1.html


30.
1. Coding: Implement last 5 min qps measure functions in O(1) time, QPS measure does not need to be exactly precise
class MeasureQPS:
def get(key):
  pass
def put(key, val):
  pass
def measure_get():
  pass
def measure_put():
  pass
2. Coding: LLVM language coding, use add and sub to implement divide (这道题本来算法并不难，就是要理解这个LLVM的语法然后快速写出来比较麻烦，提前熟悉一下可能更容易过)
3. Coding:
You are given a single URL start_url and depth value, K. Write a program that downloads all the web pages at start_url and the pages it points to up to K levels. 
The result is a directory with all the content written into it (as files or sub-directories
#
#
# Web crawler
# URL -> children URLs, up to k depth
# Possible loop, dedup and stop querying when running into visited url
Followup问了如果有url fetch时间太久要怎么办，如果在distributed system里面做，我说multithread面试官也有问具体boilterplate code会是怎么样要求写一下
‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌
4. HM round, behavioral question and project deep dive
5. System design - Design VISA network to talk to buyer and merchant's bank，
场景就是buyer去买东西刷信用卡，merchant的bank要talk to VISA network来setup和confirm，然后最后完成这笔交易。

# https://www.1point3acres.com/bbs/thread-818843-1-1.html

31. OA
1. 给一个string list，取每两个相邻string中第一个的头和第二个的尾合起来，成一个新的list。最后一个的头对应第一个的尾
2. 给一个sources list和string list，问每个string是否是sources某前几个source的concatenation
3. 蠡口 逸霸琉衣 变形，所有物体是连在一起的，问的是掉下去后最终停留位置，只需要多check一步就行了
4. destroying ho‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌uses，网上有

第二题基本是 蠡抠 姚散酒 的原题


# https://www.1point3acres.com/bbs/thread-818725-1-1.html

32. OA
第一题 连续三个数能不能构成三角形
第二题 两个string输出他们的总和 ex. "17", "36" => "413"
第三题 给定每一行的单字跟页宽, 输出要自动根据页宽换行及置中
第四题 给一串数字, 输出valid pair数, 数字长度一样且刚好差一个digit是valid
# https://www.1point3acres.com/bbs/thread-817322-1-1.html


33. OA
1. 给一个数字，计算 各个位数的乘积 和各个位数的和 的差形象管理
e.g. input: 1010 output: (0*1*0*1) - (0+1+0+1) = -2
2. 交换给定字符串：给定一个字符串，和list of sections，第一个section和第二个section位置交换，第三个和第四个交换 。。。，如果section数量为奇，最后一个section就不用交换
e.g. input:"codesignal" [3,2,3,1,1] output: "escodaignl"
3. 报纸排版，给一个vector<vector<string> paragraph（ 外面的vector代表句子数量，里面的vector存句子）和width，根据width把句子居中打印出来，超出width的话就换行。
4. towers配对，给一个数组代表towers的高度， 每次操作可以给任意一个tower高度+2或-2，计算‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌towers两两配对需要的最小操作次数，使得每队的towers高度一样
e.g. [1,4,3,2] --> [3,4,3,2] --> [3,4,3,4] 返回 2 （可配成[3,3], [4,4]两队）



34.
consumer revenue 题
followup 问 给你多一个参数表示 include 多少层的referred customer 的revenue， 让你返回。
我说用hashmap 存refer的关系 和原始 consumer revenue数据，然后recursive的相加。

# https://www.1point3acres.com/bbs/thread-816735-1-1.html


35.
看了些Databricks的面经，发现很多大多数都是这3道题其中之一，汇总如下，LZ贴了前面两道题自己的解法，也是根据地里的大神给的提示做的，
String那道题用的是BlockList，Revenue那道用 的是TreeMap

Question 1：
Design a class newString such that the time complexity for insertion, deletion and read is less than o(n).
class NewString{
public:
char read(int index);
void insert(char c, int index);
void delete(int index);
}
这道题地里看到不同的方法，很多人说用rope，以及跳表，blocklist，skipList都可以做
LZ用就是简单java，思想上和blocklist类似吧：https://leetcode.com/playground/7CSVv7ZX （仅供参考）
Question 2:
有一个系统，里面记录着每个customer产生的revenue，要你实现3个API：
1. insert(revenue): 一个新customer，产生了revenue，返回新customer的ID。customerID是自增ID，第一次insert是0，第二次是1，以此类推
2. insert(revenue, referrerID): 现有ID为referrerID的customer refer了一个新customer，产生了revenue，返回新customer的ID。这种情况下认为referrer也产生了revenue。比如说customer 0之前产生的revenue为20，他refer了新人，产生了40revenue，customer 0产生的revenue就变为60
3. getKLowestRevenue(k, targetRevenue): 给定k和revenue，要求返回>给定revenue的k个最小revenue所对应的customer ID
LZ用的是TreeMap方法：https://leetcode.com/playground/SG6ZhspC（仅供参考）
Question 3:
设计一个mockHashMap的class，其中包含这几个API：
put(key, val)
get(key)
messure_put_load()
messure_get_load()
其中put和get就和普通hashmap一样，messure方法需要返回 
average times per second that put/get function be called within past 5 minutes，就是当前时间的‍‍‌‌‌‌‌‍‍‌‌‍‌‍‌‌‍‍‌‍前五分钟内，monitor pattern
平均每秒钟put/get 被调用的次数
# https://www.1point3acres.com/bbs/thread-812809-1-1.html

36.
问了经典的 referer那道题，求大于某个range的referer. 答了用treemap但是忘了treemap是基于key的排序，就说可以加customized comparato
# https://www.1point3acres.com/bbs/thread-812344-1-1.html


37.
top k frequent in the stream
就是design一个类似wordcount的东西但是要用map reduce（naive的heap method直接跳过不给我写）

38.
题目是给定一串ip或CIDR地址，以及allow或deny两种status，以及一个ip，要求判断该ip应被allow还是deny
大概讲了一个算法，在面试官要求下优化成使用unsigned int32来表示ip。但在处理数据这一块因为用了c++不支持split，写了很多b‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌ug哈哈


39.
Design a class newString such that the time complexity for insertion, deletion and read is less than o(n).
class NewString{
public:
char read(int index);
void insert(char c, int index);
void delete(int index);
}
我当时只想到要用DoublyLinkedList+hashmap，但是这个不能解决index要前移后移的问题
后来面试官就引导我不要每个index都存，后来讨论得出差不多存index的gap是logn就行，n是可能的string的长度
比如说abfad，char是一个一个来，不用0,1,2,3,4都存map里，而是只存个0, 2, 4之类的，然后根据index离哪个近去找一个Node然后在doublylinkedlist里面把它接进去，接完之后只需要更新后面的map的value，
比如说一开始map里是{ 0: a, 2: a } 
然后insert(b, 1)，就先找到0上的a，通过a找到f, 把b接在f前面，然后把map更新成{ 0: a, 2: f }，是更新value，不是更新key，
只更新当前index和之后的位置对应的value就行了。delete也是同理的，只不过是往‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌前移。
insert和delete可能是invalid的，就throw exception，比如说当前string是abc，要insert(d, 10)就是invalid的，因为只能insert到3或者之前，
delete的index也可能过大，当然这个可能因面试官给的条件而异， 要问清楚。
我自己水平确实不太够。。。讨论了好久，面试官的hint总get不到，最后连insert都没写完，很快不出所料接到拒信><
这次详细面经分享给大家 希望大家之后碰到这个题至少别像我这么尴尬了><

# https://www.1point3acres.com/bbs/thread-798282-1-1.html

"""
// if we store the chars into lists of list
// that been said, a list with each element is a list of chars
// and assume we have number of char lists as A
// each char list size is B [bucket size]
// total string length is N
// thus we have A * B = N
// upon insert, we need to find the right char list [bucket] from buckets
// the time will be A + B (use bucket size to skip buckets util find the right one take A, insert into bucket will take B)
// upon remove the same
// the key here is to always keep each bucket in a controlable size, not let one bucket huge while others small [inbalance]
// that been said, we need to maintain the bucket in a certain size range each time after insert/delete to avoid inbalance
// and since we want each operation < O(N)
// lets say we make A = B = sqrt(N)
// thus insert/delete without maintain is O(sqrt(N))
// for maintaince, let's set the size range as sqrt(N) ~ 2 * sqrt(N) [A ~ 2 * A]
// each maintaince we check if the updated bucket needs to be split into 2 (when size > 2 * A) [take A = sqrt(N) time]
// ---LETS WRAP THE LINKEDLIST TO SAVE TIME -- -> java/kotlin linkedlist does not support connect 2 linkedlist in o(1)
// need to implement such linked list ourselfves. while using our linkedlist, the connect will be O(1)
// and loop all buckets check if need to merge small ones [take B time]
// thus total time for insert/delete are still sqrt(n)
class SubLinearString {
    private var blockSize = 0
    private var totalSize = 0

    private val blockData = LinkedList<Block>()

    fun insert(ch: Char, position: Int) {
        if (position < 0 || position > totalSize) {
            throw IllegalArgumentException("Position is out of bound.")
        }

        totalSize++

        // find position
        val foundPosition = findPosition(position)
        // insert
        val (curBlock, index) = if (foundPosition.block == null && foundPosition.blockDataIndex == 0) {
            // first element
            blockData.add(Block(LinkedList()))
            blockData.first.data.addLast(ch)

            Pair(blockData.first, 0)
        } else if (foundPosition.block == null) {
            // reach the end, append to the end at end block
            blockData.last.data.addLast(ch)

            Pair(blockData.last, blockData.size - 1)
        } else {
            foundPosition.block.data.add(foundPosition.innerIndex, ch)

            Pair(foundPosition.block, foundPosition.blockDataIndex)
        }

        // maintain size
        maintain(curBlock, index)
    }

    fun delete(position: Int) {
        if (position < 0 || position >= totalSize) {
            throw IllegalArgumentException("Position is out of bound.")
        }

        totalSize--

        // find position
        val foundPosition = findPosition(position)
        // delete
        if (foundPosition.block == null) {
            throw Exception("unexpected error! try delete but out of bound")
        }

        foundPosition.block.data.removeAt(foundPosition.innerIndex)

        // maintain size
        maintain(foundPosition.block, foundPosition.blockDataIndex)
    }



    fun read(position: Int): Char {
        if (position < 0 || position >= totalSize) {
            throw IllegalArgumentException("Position is out of bound.")
        }
        // find position
        val foundPosition = findPosition(position)
        // get char
        return blockData[foundPosition.blockDataIndex].data[foundPosition.innerIndex]
    }

    fun printToString(): String {
        println("==blockSize - $blockSize==")
        return blockData.map { block ->
            block.data.joinToString("")
        }.toList().joinToString("|")
    }

    private fun findPosition(position: Int): Position {
        var remain = position
        var curBlock: Block? = null

        var mover = 0
        val iter = blockData.iterator()
        while (iter.hasNext()) {
            curBlock = iter.next()

            if (remain < curBlock.data.size) {
                return Position(
                        blockDataIndex = mover,
                        block = curBlock,
                        innerIndex = remain
                )
            }

            mover++
            remain -= curBlock.data.size
        }

        // reach the end
        return Position(
                blockDataIndex = mover,
                block = null,
                innerIndex = 0
        )
    }

    private fun maintain(curBlock: Block, blockIndex: Int) {
        blockSize = Math.sqrt(totalSize.toDouble()).toInt()
        // check if it needs to be split
        val doubleSize = 2 * blockSize
        if (curBlock.data.size > doubleSize) {
            val newBlock = Block(LinkedList())
            newBlock.data.addAll(curBlock.data.subList(doubleSize, curBlock.data.size))

            while (curBlock.data.size > doubleSize) {
                curBlock.data.removeLast()
            }
            blockData.add(blockIndex + 1, newBlock)
        }

        // check if it needs to be merged
        val blockIter = blockData.iterator()
        var block = if (blockIter.hasNext()) blockIter.next() else return

        while (blockIter.hasNext()) {
            val nextBlock = blockIter.next()

            if (block.data.size + nextBlock.data.size < blockSize) {
                block.data.addAll(nextBlock.data)
                blockIter.remove()
            } else {
                block = nextBlock
            }
        }
    }
}

data class Block(val data: LinkedList<Char>)

data class Position(
        val block: Block?,
        val blockDataIndex: Int,
        val innerIndex: Int
)
"""


40.
credit card system design
经典design题目 主要说下validation和avoid double transaction即可
# https://medium.com/airbnb-engineering/avoiding-double-payments-in-a-distributed-payments-system-2981f6b070bb

# https://www.1point3acres.com/bbs/thread-807718-1-1.html

41.
报一个地里原题的数据点：https://www.1point3acres.com/bbs/thread-803679-1-1.html。设计 key value s‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌tore 能统计 QPS。
follow-up：如果允许一定的误差，能否优化时间空间复杂度，欢迎讨论。
# https://www.1point3acres.com/bbs/thread-805991-1-1.html

42.
面经：地里已经很多了 Cusotomer Revenue那一道 (https://www.1point3acres.com/bbs ... ighlight=databricks), 
这里多了一问如果refer的customer之间算connected，给一个customer id和最大depth问在这个depth内从id出发的customer revenue sum。主要提醒一下光准备题目怎么写是远远不够的！其实60分钟的面试中写题的时间是非常短的，大部分时间都在讨论各种数据结构之间的trade-off。
比如说在这题里如果是insert-heavy 怎么办，read-heavy怎么办， 如果是read, writemockHashMap都想注重怎么办？假如说我们用一个hashmap做背后的data structure使insert和read很快，这个时候get_K_customer_above_threshold怎么做可以最优化？如果最后一问里我们用graph来做且愿意牺牲insert的run time来换取get customer revenue sum的速度应该怎么办？ 各种run-time complexity问的非常细。

比如说我一直以为build a heap的run tim‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌e是O(NlogN) (其实是O(N))，

# https://www.1point3acres.com/bbs/thread-804106-1-1.html


43.
设计一个mockHashMap的class，其中包含这几个API：
put(key, val)
get(key)
messure_put_load()
messure_get_load()
其中put和get就和普通hashmap一样，messure方法需要返回 average times per second that put/get function be called w‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌ithin past 5 minutes，
就是当前时间的前五分钟内，平均每秒钟put/get 被调用的次数

# https://www.1point3acres.com/bbs/thread-803679-1-1.html


44.
非常友好的国人面试官（如果您看到了这贴，thx！）
10min 自我介绍，过简历，上一次实习的内容
题目：设计一个key-value存储系统，并且支持一个统计5分钟内get/set的次数的API
我面试的时候大脑不太清醒，还是被捞了。。最后用中文聊了15分钟

题目：设计一个系统记录客户的付款。系统维护一个从0递增的id，每笔付款都要返回这笔付款的id。付款函数pay(val, referrer)还接受一个optional的referrer，表示这笔钱的referrer的付款id。接下来，实现一个函数get(k, thresh)，返回所有“关联金额”大于thresh的付款中，“关联金额”最小的k笔付款的id。“关联金额”的意思是这笔付款的数目+所有直‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌接被我refer的付款的数目。
follow-up：修改“关联金额”的定义，改成直接或间接（但是距离在若干次refer以内）被我refer的付款。
体验：这个姐姐真的非常的有趣，有一种高智商人群自带的幽默感。可以说是最近体验最好的一次面试了。
还有一轮manager吹水，但是还不太理解databricks的商业模式，所以不打算接offer。。
# https://www.1point3acres.com/bbs/thread-800479-1-1.html

45.
刚面了onsite前两轮，第一题是multithreading web crawler，
第二题是实现一个commandline interface的SQL，要求能完成from, select, take, orderby, join, countby这六个操作。

第一题截屏附上。等后两轮面了再来发帖子，有大佬知道大概的话也麻烦透露一下啦，新‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌人看不了面经qaq。
# https://www.1point3acres.com/bbs/thread-800423-1-1.html

46.
设计一个和customer/revenue有关的系统：
前三问与地理一致，可以参考这个链接 https://www.1point3acres.com/bbs/interview/databricks-software-engineer-775327.html
第四问貌似没见到有帖子提到过，附上了截图。

这道题前三问在之前的帖子里就有很多讨论，lz在准备的时候看到很多朋友建议用treemap（Java）或者maintain一个sorted list/doubly linked list （Python）。因为lz一直用的是python，python里面也没有类似treemap的数据机构，所以在面试的时候我用了一个Sorted List, 每次insert/refer的时候都用binary search tree找到对应的index进行insert。最后找k个大于threshold的值也适用binary search找到sorted list中大于threshold 的值。
当时我写完之后面试官表示looks good，但之后交流下来感觉面试官认为这道题最好的数据结构还是需要用一个balanced tree，每次insert/delete/找threshold的时候找到树中对应的位置进行操作（类似于Java中的treemap的底层实现）
第四题是求从id开始一共走max_nesting步可以得到多少总的revenue。其实就是一个很简单的BFS，首先记最终return值为total_rev, 只需要从id开始，对于每一个可以达到的new id（已经在前三问‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌中store了），将total_rev加上new_id的revenue 即可。
以上是一些个人想法，如果有错误欢迎大家在评论区指正。最后，如果有帮助请一定加米，新人账户急需米看贴，谢谢！

# https://www.1point3acres.com/bbs/thread-798860-1-1.html


47.
第一题 在一个从0-n个int中，找出含有0，2，4这三个digit的总和
第二题
第三题 输入格式["INSERT code"] ["INSERT signal"] output: codesignal
模拟INSERT，DELETE，COPY，PASTE操作，DELETE是删除当前text的最后一个字符，INSERT是插入其后的text,COPY<Index> 是从当前text ‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌index位置开始进行复制，复制到一个clipboard（得自己定一个）. PASTE粘贴text
第四题 找到灯光最亮的index最小的点（地理原题

# https://www.1point3acres.com/bbs/thread-798218-1-1.html


48. OA
对应李扣的 是 要三二九 和 四十八
# https://www.1point3acres.com/bbs/thread-797091-1-1.html

49.
Round 1 Coding: LC 三霸凌 但是要RandomizedHashMap
Round 2 Coding : 类似于这个 https://www.geeksforgeeks.org/de ... m-in-constant-time/ 要用ROPE 不知道的人 我觉得这题没办法过
Round 3: BQ go over projects in resume
Round 4: Design payment 接口。这题开始理解的有些问题，design了一个internal 的paymen‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌t system，然后面试官说只要和VISA的接口API设计。再改时间就有点来不及了。
最后的争议点在idempotency的数据结构是什么，id怎么生成。

# https://www.1point3acres.com/bbs/thread-793646-1-1.html

"""
// use a key to index and index to key map to track the index of key 
// and for the convenience of quick lookup
// when random just lookup in indexToKey with random
// when delete swap the last index's key with deleted key's position to make it o(1)
class RandomHashMap<K, V> {
    private val internalMap = mutableMapOf<K, V>()
    private val keyToIndex = mutableMapOf<K, Int>()
    private val indexToKey = mutableMapOf<Int, K>()
    private val random = Random()

    fun get(key: K): V? {
        return internalMap[key]
    }

    fun put(key: K, value: V) {
        val existed = internalMap.containsKey(key)

        internalMap[key] = value
        if (existed) return

        val index = keyToIndex.size
        keyToIndex[key] = index
        indexToKey[index] = key
    }

    fun delete(key: K): Boolean {
        if (internalMap.remove(key) == null) {
            return false
        }

        // swap the last index to the current key's index
        val lastIndex = indexToKey.size - 1
        val lastKey = indexToKey[lastIndex]!!
        val currentIndex = keyToIndex[key]!!

        indexToKey[currentIndex] = lastKey
        keyToIndex[lastKey] = currentIndex

        indexToKey.remove(lastIndex)
        keyToIndex.remove(key)

        return true
    }

    fun random(): K {
        return indexToKey[random.nextInt(indexToKey.size)]!!
    }
}
"""


50.
设计数据结构insert(int index, char val)
delete(int index)
read(int index)
跟面试官讨论了可以使用的数据结构，并且讨论每种数据结构实现起来的时间复杂度，
最后锁定一种，思路可以是BST也可以是 b‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌ucket list，然后要求把代码实现出来。
# https://www.1point3acres.com/bbs/thread-789555-1-1.html


51.
// 1. system & design，还是web crawler那个题，先写个单线程的限制最大depth的BFS的crawler，很快写完，然后开始讨论performance瓶颈。
// 一开始提供了三个utility function：
// (1）fetch(URL: url) -> HTML: content; 
// （2）parse(HTML: content) -> List<URL>; 
// (3) save(URL: url, HTML: content)。
// `save`是把数据存在disk上，`fetch`是发个network request，
// `parse`是in-memory解析html page，
// 有一个要求就是每一次crawl的URL都要调用save存在local disk上面。
// performance瓶颈的讨论就集中在这3个函数上，问到了每个函数的大致的latency的范围，这个代码的CPU利用率会不会高，
// 如果有bottleneck大概率在那儿（应该是在network上），问完了就到followup了，怎么提高速度，那就多线程操练起来吧。
// 楼主因为Java用得多，就拿ForkJoinPool写了一个solution，面试官好像不怎么了解Java的ForkJoin framework，所以问了很多关于这个framework怎么做scheduling的，有没有blocking wait之类的问题，
// 花了十几分钟讲ForkJoinPool的cooperative scheduling的原理，幸好之前看了不少文章。问到最后感觉面试官并不怎么满意，大概率这轮挂了。

// 2. tech fit，对面问了15分钟的关于我的过去做的project的问题，然后就开始coding了。题目是实现一个hashmap，但是支持一个额外的API是能够随机返回一个hashmap里面的value，不过概率上要是uniformly distributed那种。举了个栗子：put("a", "a"), put("b", "b"), put("c", "c"), put("d", "c")，这个时候map的values里面有1个"a"，1个"b"，2个"c"，那么就应该有50%的概率返回c，25%的概率返回a或b。其实就是蠡口三巴林的变种，
// 不过楼主当时脑子不清醒，走了弯路，最后才被面试官带回来，勉强写完。这轮看起来应该也挂了。
// 3. BQ，对面是个director，没啥特别的，正常套路。神奇的是中间对面突然问楼主现在有没有在面别家，现在是什么状态了啥的，一时不知道啥意思，
// 就如实回答了有面别家，然后有几个offer不过还没deadline。现在想想不知道这里面有没有坑。。。。。
// 4. coding，这轮是经典的LLVM，写完后就讨论了下负数除法怎么做。
// 5. Architecture & Design，还是那个payment network题，之前的面经都没咋提到细节，
// 这题面试官明确说了就讨论payment authorization这个scenario，fund settlement不讨论。于是就从数据本身开始，到API interface design，
// 再说到unique id generation（for transaction id），idempotency的实现，
// failure han‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌dling（retry，error code分类，可重试error和不可重试error，etc.)，最后讨论了一下scalability，
// 存transaction info的DB怎么scale，需要考虑什么因素，甚至问道了预估这个payment的transaction的volumn有多大，
// 楼主瞎猜说对Visa这种全球的payment network来说怕是得上1M/s吧，对面表示呵呵，其实全美就几千个QPS，
// 全球也不会多多少因为visa的市场主要就是在美国。。。。。。楼主心想就这么点QPS就不用担心scalability问题了吧，
// 就提了一下这种流量的write traffic普通的数据库都能handle，只需要保证把API server和数据库都geo distributed到有business的地方就好了。P
// ayment authorization flow可以提前看看http://blog.unibulmerchantservic ... yment-system-works/，
// 里面有提到这个背景，面试官画的图基本就是这个link里面描述的authorization flow。

# https://www.1point3acres.com/bbs/thread-787263-1-1.html


52.
店面是insert delete get random
1. Customer API的面经题。跟这个一样 https://www.1point3acres.com/bbs/thread-704860-1-1.html
问了一个followup，是让实现  get_nesting_level(int customer_id, int nesting_level)
比如 get_nesting_level(1, 0) -> 返回 customer_id=1 的自己的revenue
get_nesting_level(1，2) -> 返回custoemr_id=1, 同时 包括他refer的两层的结果。举个例子 1 refer 2, 2 refer 3， 那么这里就要return 1, 2, 3 的总和。
面试官想optimize这个function，牺牲insert的性能。我的解法就是维护一个Map<CustomerId, Map<NestingLevel, Revenue>> 的map，insert的时候一直recursive update refer的人，比如insert customer id =3 的时候，要update 2 和1 。
2. LLVM。 地里面看了一圈，这题不用想太复杂了，面试的时候只让用sub add 那几个instruction，不用leetcode拿到题 做bitwise shift。所以就写了一个简单的 一直减的loop
3. Web Crawler。 这题说是design，其实主要在写代码。
题目问的是single machine 怎么从root_url爬三层。 问了怎么crawler能停止，如何去重，如何failure recovery
4. Visa payment network.
5. Manager behavior 国人manager， 人很好，没问啥harsh的。
面完之后hr跟我说debrief后, u did very well. Strong positive results across all the interviews, feedback was solid positive. really nice job. Several interviewers were really impressed with your ability to walk through the problems, communicate effectively and provide optimal solutions. very excited to proceed forward.
然后发takehome， 我就随便做了下（最后贴一下takehome吧，这个没看到别人写过，写的比较弱鸡）
后来是聊team，聊了好几个组，找了一个比较硬核的。
然后就是要reference，说是找manger 和TL比较好，我觉得找前任manager这种有点太勉强了吧，找了几个之前的前同事，结果都是hm去打电话，问怎么认识这个人啊，是你work过的前百分之几啊。因为我找的都是熟人嘛，朋友都说各种吹我，都说稳了。
最终refernce collect 完了，距离onsite也两周了，然后送HC，结果hr回复就很慢了，HC收先要了一下推荐人的linkedin。又等了两天，最后hr跟我说HC要reject。。。
Bar高我能理解，因为我也不是infra出身，个人觉得自己实力也不强，只不过onsite完了听hr口气是觉得自己面的还挺好的。reference的时候我也是到处联系人，最终给拒了，也是花了好多精力和时间。耽误自己不说，reference也要和hm约时间打电话。。。我只是想吐槽，如果一开始HC觉得我case不strong，能不能onsite完就把我拒了，何必让我做take home 以及team match呢？
# https://www.1point3acres.com/bbs/thread-781585-1-1.html


53.
店面：原题，类似merge list
onsite 4轮
1. LLVM：发挥还不错
2. Behavior：发挥还不错
3. KV Store：这一轮面完感觉还可以，因为看了很多面筋，准备还可以，包括WAL，读写锁，lock sharding等，
大概30分钟不到就完事了（除去聊天问问题），后面就这一轮negative，被抓了几个小漏洞，怎么说呢，有2个是我回答的仓促了，当然，有1个文件操作确实答得不好。
4. snapshot list：发挥还不错，也是原题
后面让我加面了一轮系统设计，也是原题，和LC的多线程 BFS 爬虫差不多，顺利过关。
homework也花了好久认真写完通过了。

# https://www.1point3acres.com/bbs/thread-781036-1-1.html

54.
基本上是蠡口肆意，区别是array里面是non-negative integer，需要找到最小的缺失的non-negative integer，感觉是蠡口那个题的简化了一点的版本。
楼主之前没刷过这个题，一开始想了一个简单粗暴的用set存数字然后从0开始挨个扫描看哪个缺了，写完了面试官说你想想能不能不用额外空间，于是我就卡住了。
然后面试官提示可以更改input，于是想到了把array index当作hash table的key的思路，最后写了一个不停swap的解法，中间出了一个bug，
面试官给了个test case来提示的，然后很快改掉了。这时候基本就没时间，剩5分钟了，面试官又问了怎‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌么把input还原回去，当时实在想不出什么，
回来看了蠡口上的解法，感觉面试官就想朝蠡口上的解法引导。面完以为挂了，结果第二天早上收到邮件说准备Hiring Manager面试了，感觉被国人大哥抬了一手，感恩！
# https://www.1point3acres.com/bbs/thread-779575-1-1.html

"""
# The key here is to use swapping to keep constant space and also make use
# of the length of the array, which means there can be at most n positive integers. 
# So each time we encounter an valid integer, find its correct position and swap. 
# Otherwise we continue.
#  ignore all the negative, > n
#  put the other value back to its order position A[A[i]-1]
fun findNonNegativeMissingInteger(nums: Array<Int>): Int {
    if (nums.isEmpty()) {
        return 0
    }

    var mover = 0
    while (mover < nums.size) {
        // if current slot does not match its num, swap to the slot it belongs to
        // ex. [1, 4, 2]
        // mover = 0
        // nums[0] should be 0 but is 1
        // thus swap nums[0] <> nums[1]
        if (
                nums[mover] != mover 
                && nums[mover] >= 0 
                && nums[mover] < nums.size 
                && nums[mover] != nums[nums[mover]]
        ) {
            // swap
            val temp = nums[mover]
            nums[mover] = nums[nums[mover]]
            nums[temp] = temp
        } else {
            mover++
        }
    }
    
    for (i in nums.indices) {
        if (nums[i] != i) {
            return i
        }
    }
    
    return nums.size
}
"""


55.
地里原题。有一个系统，里面记录着每个customer产生的revenue，要你实现3个API：
1. insert(revenue): 一个新customer，产生了revenue，返回新customer的ID。customerID是自增ID，第一次insert是0，第二次是1，以此类推
2. insert(revenue, referrerID): 现有ID为referrerID的customer refer了一个新customer，产生了revenue，返回新customer的ID。这种情况下认为referrer也产生了revenue。比如说customer 0之前产生的revenue为20，他refer了新人，产生了40revenue，customer 0产生的revenue就变为60
3. getKLowestRevenue(k, targetRevenue): 给定k和revenue，要求返回>给定revenue的k个最小revenue所对应的customer ID
第三问比较tricky，我现在也不知道什么算是对的，就po下我的思路就当抛砖引玉了。最直观的肯定是用heap，复杂度是O(N + klogN) 就是建堆+取k个数。
但我觉得如果第三个API call的远比insert的次数（customer个数）多，可能用array + binary search 会比较好，这样call第三个API的时候你可以用binary search找到targetRevenue的index然后直接取后面k个revenue对应的ID，只要O(logN)。
虽然这样你insert的时候需要一直maintain array sorted，
但其实每次你只需要往‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌sorted array里insert两个数就好（新customer的revenue和referrer的updated revenue)，整体复杂度是O(N^2)，在call第三个API特别多的时候这个反而是更快的。^^以上N为总客户个数。

# https://www.1point3acres.com/bbs/interview/databricks-software-engineer-775327.html

56.
string delete(), insert(), read() 需要 sub linear 最好space efficient
# https://www.1point3acres.com/bbs/thread-778392-1-1.html

57.
第一轮，美国小哥，给了一道高频题。 给ip_cidrs以及对每一个ip_cidr的policy(allow/deny). 写一个函数，输入是ip，输出是true/allow or false/deny。 （当有多个match的时候，返回第一个结果。 如果没有match，返回true）
第二轮，hr说挂在这里了。华人大哥。高频。Kv store with data persistent functionlaity。两个方法- WALs and snapshot。 需要写代码（不用run）。
第三轮，随便聊了聊。没有题目。
第四轮，llvm。原理。做divide。 我就一直减直到小于0。
第二轮挂的莫名其妙。都自己说出来了。

# https://www.1point3acres.com/bbs/thread-776088-1-1.html

58.
一个是query value大于某个threshold的最小的k个值和对应id...
感觉是俩做法
如果是insert多 就在query的时候用个priority queue？
如果是q‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌uery多 就直接维护一个链表然后线性找？

# https://www.1point3acres.com/bbs/thread-775191-1-1.html

// ==<HERE>==
59.
 coding - remove sql comment
* tech deep dive
* llvm
* system design - web crawler
* architecture - visa network
* HM BQ
* 回家作业: sql query over csv

* coding 这道题目在论坛里被提到过多次了， 没啥特别的， 比 LC 722 简单一点，但最好自己先练一下。
* Tech deep dive - 就是聊聊楼主过去的项目，聊了大约40分钟， 对方表示问不出问题了，于是楼主问了几个软球问题。
* LLVM - 做减法，不需要处理 int.min, 和除以0的情况。准备过的同学估计5分钟就搞定了，这轮主要考的是演技。
* system design - 楼主遇到了之前在论坛里被提到过的一位香港同胞，goofys 的作者。在另外一个帖子里，那位大哥被港弟给坑了。
楼主觉得大概率不会换题，所以也没多虑，就没绕道。虽然感觉此人确实态度有点傲慢，不够礼貌，但由于楼主准备充分，3阶段的代码也顺利写出来了（
1， 普通 BFS 爬， 2， 限制最大深度 BFS 爬， 3， 多线程加速爬），楼主用的是某一个单线程的脚本语言，
所以伪代码里用了 async await + fork process。这题稍微看一下就没啥难度，楼主也搞不清这里面能考察出多少 system design 的本领。
港男见楼主迅速写完代码，开始各种发问，稍微有点 OS 常识即可一一化解。40分钟左右，港男也不能问出更多有意义的问题了，提早结束。
* HM BQ - 没啥好多讲的，和 tech deep dive 类似， 也是介绍项目的经验，讲故事就完事了。
* 架构设计， 聊 visa network 的设计。不清楚是不是梁静茹给了他们勇气来问这题， visa 上个世纪很早就构建出了他们的 payment network, 大量使用大型机。
这个搞 spark 的小厂难道觉得他们很懂支付？这题楼主认为能聊的点并不多，支付系统需要强一致性，从 visa 的角度来讲，其实并不能玩出太多花样来，
和普通互联网应用有很大的区别。我不知道砖厂的工程师是如何来评判面试者的设计的，就是觉得从专业的角度来说，他们可能也不懂。
* onsite 结束以后，第二天拿到了回家作业。实现一个简单的 sql query engine against csv data, 主要就是写个简单的 parser, 处理数据注意 edge case 即可，代码则要注意工整和测试完整性。 楼主利用周末完成了作业， 写的比平时干活还认真。
结局：
* 背景： 楼主在一线大厂，L6 也很久了。
* 过了几天以后，最后被告知 HC 不愿意给楼主 L6, 只肯给 L5。追问原因，说虽然所有的轮次都拿到了 hire, 但是 system design 他们没有能够拿到足够的 signal 来支撑 L6。楼
主这才反应过来被港男阴了，大意了，没有闪。说实话楼主不觉得这个破题能体现出 5 和 6 的差距，楼主平时在厂里拧的螺丝要比这个复杂太多了。
* 问猎头是不是可以加面一轮 system design 证明楼主还‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌是牛逼的，猎头表示， HC 说可以， 但是要等6个月才行。 -_-

# https://www.1point3acres.com/bbs/thread-773542-1-1.html
"""
// inline comment
// comment in quote does not count
// quote could be escaped
class SQLCommentRemove {
    fun removeComment(sqls: String) {
        if (sqls.isEmpty()) {
            return
        }

        var index = 0
        var isInComment = false
        var isInQuote = false
        while (index < sqls.length) {
            // check if new line
            if (sqls[index] == NEXT_LINE) {
                // if not in quote, reset inline comment
                if (!isInQuote) {
                    isInComment = false
                }
            }
            // if not in comment
            if (!isInComment) {
                if (isInQuote && (sqls[index] == SINGLE_QUOTE) && !checkIsQuoteEscaped(sqls, index)) {
                    // check if end quote when it is in quote
                    isInQuote = false
                } else if (!isInQuote && (sqls[index] == SINGLE_QUOTE)) {
                    // check if in quote when it is not in comment
                    isInQuote = true
                }
                // check if in comment when it is not in quote
                isInComment = !isInQuote && checkIsComment(sqls, index)
            }

            if (!isInComment) {
                print(sqls[index])
            }

            index++
        }
    }

    private fun checkIsComment(sqls: String, index: Int): Boolean {
        if (index == sqls.length - 1) {
            return false
        }

        return sqls.substring(index, index + 2) == COMMENT_STR
    }

    private fun checkIsQuoteEscaped(sqls: String, index: Int): Boolean {
        if (index == 0) {
            return false
        }

        var quoteToken = sqls[index].toString()
        var mover = index - 1

        while (mover >= 0) {
            if (sqls[mover] == ESCAPE_CHAR) {
                quoteToken = sqls[mover] + quoteToken
                mover--
            } else {
                break
            }
        }

        // java str as \' or \\\\\' sql str as ' or \\'
        if (quoteToken.length % 2 == 1) {
            return false
        }
        // java str as \\\' sql str as \'
        return true
    }

    companion object {
        private const val NEXT_LINE = '\n'
        private const val SINGLE_QUOTE = '\''
        private const val ESCAPE_CHAR = '\\'
        private const val COMMENT_STR = "--"

    }
}
"""


60.
原题是给一个string 要求实现高效的三种操作：get(int key), insert(char c, int key), delete(char c, int key)
面试的时候没做好反正，后来回来研究了一下，发现有两种做法，一种是Rope，还有一种是块状数组。复杂度大概是sqrt(N)。不过感觉一个电面45分钟根本不可能写的完。可能有别的‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌更好的算法。
# https://www.1point3acres.com/bbs/thread-771881-1-1.html

61.
第一轮店面，就一道题。Given a list of firewall rules, such as
[{"192.168.1.22/24": allow}, {"192.168.122.126/20": deny}, ...]
Write a function to determine if an IP is allowed to pass the firewall.
Something like:
boolean canPass(List<Rule> firewall, String ip) {}
‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌店面后安排了做一个项目，提交后说，第二天收到recruiter 说not move forward this time。。。
# https://www.1point3acres.com/bbs/thread-770137-1-1.html

62.
10min BQ + 45min 高频 意思流
不是直接给题目，是从一个基本场景引导你 自己想出cache 然后再说出几种cache的类型，以及使用场景，最后实现

# https://www.1point3acres.com/bbs/thread-768737-1-1.html


63.
* 面经插入删除字符
* 经理行为：最大挑战，得到的负反馈，处理争议，做艰难决定，上次升值，现在的级别，到下个级别的茶具
VO：
* 设计: 单机键值，先写日志，加锁，并发，写伪代码
* 新语言
* 设计: 维萨支付，问清需求，算好系统规模和资源
* 经理行为: 与店面相似，加问怎么帮助组员成长，做季度规划，处理优先级
* 领域设计：前半部分问目前负责的系统架构，深入问调度算法及具体实现，后半部分设计用于数据仓库的ETL发布订阅系统
* 读论文：提前一周发某数据库顶会论文；问基本概念，需要解决的问题，主要的论文想法和贡献，论文方案的好处坏处，怎么在现有系统上实‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌现论文方案。
最后两轮是六级加面，无带回家作业
# https://www.1point3acres.com/bbs/thread-762689-1-1.html

64.
// 第一轮，architecture，设计visa payment system。注意handle double payment和保存payment状态，参考stripe的idempotent api设计。
// 第二轮，llvm
// 第三轮，manager bq
// 第四轮，tech fit。一个ABI 从头到尾不问问题，只会一句make sense。
// 第五轮，design web clawer， 一个貌似来自hk的人面的。这个人非常push and aggressive，你如果写错一个地方被他抓住就一直问不放手。
// 虽然这一轮挂了有我自己的问题，但是面试体验很不好。我问他如果你觉得bfs写法有问题，那我需要写dfs吗？
// 我觉得dfs更不合适这个问题，他回一句：you are interviewing, not me right? 
// 正常人不应该提升一下你bfs哪里写的有问题吗？这个人是goofys 的作者，大家碰到他建议reschedule。
// 第六轮，ip cider，给定一个list of rules [(cidr1, allow), (cidr2, deny), (cidr3, allow), ...],  （cidr 请参考刷题网751），
// 和一个 ip，判断这个ip是让过还是不过。
// 这题最恶心的地方是rule之间会有冲突，比如第一条rule 算过，第二条rule算fail，以最后一条fail的为准。这个地方把我坑了，
// 我问它是不是first match win，它说是的，结果它给贴的test case是last match win。害我debug 巨长时间。
// 更恶心的是还剩20分钟不到的时候code pad 挂了，一刷新code没了而且连不上。导‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌致第一问写完没跑对test case，第二问没写。
// 第二问是把ip换成一个range，让你判断过还是不过。这个interviewer最后提交的feedback是我第一问没写完，
// 但是也没说code pad断了，更没说它给的clarification有问题。一位赵姓的new grad女生在db干了三年，新手interviewer，不懂给提升，
// clarification也给错，feedback写的很片面，建议避开。
// db在我面的这些公司里是面试体验最差的，也是我唯一一家挂的。interviewer水平除了最后一位其实都还不错，我拿offer的那几家interviewer水平都不太如我。

// # https://www.1point3acres.com/bbs/thread-761408-1-1.html


65.
// 類似李口 二三
// 寫兩個classIterator, mergeIterator
// 分別都要有 next(), hasNext()
// it = Iterator([1,2,3])
// it.next() => 1
// it.hasNext() => True
// it1 =  Iterator([1,4,6])
// i2=  Iterator([2,3,5])
// mi = mergeIterator([it1, it2])
// mi.next() => 1
// mi.next() => 2
// mi.next() ‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌=> 3
// mi.hasNext() => True
// 要求時間最低
// 一定要用 heap/pq 解
# https://www.1point3acres.com/bbs/thread-759976-1-1.html

66.
实现insert(), delete(), read()，要求时间复杂度都是线性以内。
最优解是用Rope.我用了blocking list实现。写了代码，没有要求跑。

# https://www.1point3acres.com/bbs/thread-757029-1-1.html


67.
面经攒人品～自己海投的～
API:
// public int insert(int number) --> 插入一个新的顾客
// public int insert(int number, int referralId) --> 插入一个顾客和它的推荐
// 一个顾客的rev = 自己的rev + 直接推荐客户rev
// For example,
// insert(100)  -> customer0, rev 100
// insert(200, 0) -> customer 0, rev 300=100+200, customer1, rev 200
// insert(150, 1) -> customer 0, rev 300=100+200, customer1, rev 350=200+150, customer2, rev 150
// API:
// public int[] getKCustomerRevenueBelowThreshold(int threshold) 返回rev小于指定threshold的客户id, 要求按‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌rev从大到小排列 -->  getKCustomerRevenueBelowThreshold(200), return [2, 0]
面试官希望的答案：维持一个客户利润由小到大的数组，插入二分查找，利润小于threshold二分查找。。
# https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=702466


68.
// 上来先解释map-reduce是什么，通过最简单的例子来说明。
// 正式开始，给了一道map reduce相关的题目：
// # Given a large number of key-value pairs and a small number k (< 1,000,000), 
// implement a MapReduce algorithm to find the k largest values associated with each key.

// # For example, with k = 2 and the following input key-value pairs
// # ("a", 2L)
// # ("b", 3L)
// # ("c", 1L)
// # ("a", 1L)
// # ("b", 2L)
// # ("a", 3L)
// # the expected output is
// # ("a", 2L)
// # ("a", 3L)
// # ("b", 3L)
// # ("b", 2L)
// # ("c", 1L)
// 解释map跟reduce分别要做什么？
// 多少个mapper最优？
// 我的回答： map function里面用hashmap存key -> list of values in descending order (at most k value)，然后在reduce function里面对所有同一个key的value lists，类似merge order list的方式取出前k个。
// 实现 reduce function 里面 top_k() 的部分。讨论时间空间复杂度

// Followup 部分: if k is very large, basically you can't store all K values in memory, what do you do?'
// 这部分根据提示，思路大概是这样：
// 如果你知道the ‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌k-th largest number的值，你就可以直接输出大于等于他的value。所以问题第一步，是怎样找到 k-th largest number。
// 而想要找到k-th largest number，提示是：能不能想办法推断k-th largest number 是正数还是负数。其实是可以的，只需要统计正数的个数。如果正数的个数大于k，那么k-th largest number 必然是正数。所以按照这个思路其实就类似于二分查找，
// 第一轮先找到max value 和min value, 第二轮如果大于mid value的count > k，那么k-th largest必然大于mid，所以下次二分查找就在mid ～ max的范围里面找。

# https://www.1point3acres.com/bbs/thread-752811-1-1.html


69.
第一轮 2位印度小哥。都很nice。题目是常见的revenue的那题。要求实现insert(revenue), insert(ID, revenue), getKLowestRevenueHigherThanThreshold(K,threshold) 这3个API。讨论讨论时间复杂度。follow up。如果revenue计算是K级深度的话怎么变。要求实现了新的Insert。然后口述getK的变化。

第二轮 系统设计。多线程爬虫。问的很细。但是给的引导也很足。

第三轮 LLVM 面试官帮我省去了几乎所有的corner case。不用考虑负数和0。linear 复杂度就够了。。也是很nice

第四轮 BQ。背景很牛逼的一个大叔。之前在PLTR。多年startup的经验。也是很nice

# https://www.1point3acres.com/bbs/thread-750583-1-1.html


70.
merge iterator
有follow up问多线程怎么改，还有数据量很大的话要怎么改

# https://www.1point3acres.com/bbs/thread-749150-1-1.html

"""
class SortMerger(
        iterators: List<Iterator<Int>>
) {
    private val minHeap = PriorityQueue<IteratorEntity> { o1, o2 -> o1.curValue - o2.curValue }

    // o(n)
    init {
        iterators.forEach { iter ->
            if (iter.hasNext()) {
                minHeap.offer(IteratorEntity(iter, iter.next()))
            }
        }
    }

    // o(1)
    fun hashNext(): Boolean {
        return minHeap.isNotEmpty()
    }

    // o(logN)
    @Synchronized
    fun next(): Int {
        val cur = minHeap.poll()
        if (cur.iterator.hasNext()) {
            minHeap.offer(IteratorEntity(cur.iterator, cur.iterator.next()))
        }

        return cur.curValue
    }
}

data class IteratorEntity(val iterator: Iterator<Int>, val curValue: Int)

"""


71.
// 经典的面试题 string的操作
// insert(char s, int index)
// remove(int index)
// get(int index)
// 这个问题我看其实好多人都已经在这里报过了，但是没有什么人给出一个特别好的solution，这边我给一下我的想法

// 斗胆发一个自己写过的
// 两周前店面pass

// # class BlockList {
// #     struct Block {
// #       vector<char> data;  
// #      };
// #     typedef list<Block>::iterator blockIter;
// #     typedef vector<int>::iterator dataIter;
// #     list<Block> blockList;
// #     int totalSize = 0;
// #     int blockSize = 0;
// #     public :
// #     void insert(char c, int pos){
// #         totalSize ++;
// #         blockSize = sqrt(totalSize);
// #         if (blockList.empty()) {
// #             auto iter = blockList.insert(blockList.begin(), Block());
// #             iter->data.emplace_back(c);
// #          } else {
// #             auto iter = find(pos);
// #             if (iter == blockList.end()) {
// #                 blockList.back().data.emplace_back(c);
// #             } else {
// #                 iter->data.insert(iter->data.begin()+pos, c);
// #             }
// #         }
// #         maintain();
// #     }
// #    
// #     void maintain(){
// #         // split bigger
// #         // merge smaller
// #         for(auto iter = blockList.begin(); iter != blockList.end(); iter ++) {
// #             if (iter->data.size() > 2 * blockSize) {
// #                 Block b;
// #                 b.data.assign(iter->data.begin(), iter->data.begin() + blockSize);
// #                 blockList.insert(iter, b);
// #                 iter->data.erase(iter->data.begin(), iter->data.begin() + blockSize );
// #             }
// #         }
// #         for(auto iter = blockList.begin(); iter != blockList.end(); iter ++) {
// #             auto nextIter = next(iter);
// #             if (nextIter != blockList.end() && iter->data.size() + nextIter->data.size() <blockSize) {
// #                 iter->data.insert(iter->data.end(), nextIter->data.begin(), nextIter->data.end());
// #                 iter = blockList.erase(nextIter);
// #             }
// #         }
// #     }
// #    
// #     blockIter find(int& pos) {
// #         int sum = 0;
// #         for(auto iter = blockList.begin(); iter != blockList.end(); iter ++) {
// #             sum += iter->data.size();
// #             if (sum>pos) {
// #                 pos -= sum - iter->data.size();
// #                 return iter;
// #             }
// #         }
// #         return blockList.end();
// #     }
// #    
// #     void erase(int pos){
// #         auto iter = find(pos);
// #         if (iter != blockList.end()) {
// #              totalSize --;
// #              blockSize = sqrt(totalSize);
// #             iter->data.erase(iter->data.begin() + pos);
// #         }
// #         maintain();t
// #     }
// #     char get(int pos) {
// #         auto iter = find(pos);
// #         if (iter == blockList.end()) return '.';
// #         else return iter->data[pos];
// #     }
// #    
// #     void print() {
// #         for(Block b : blockList) {
// #             cout << "| " ;
// #             for (char c : b.data) {
// #                 cout << c << " ";
// #             }
// #         }
// #      cout <<endl;
// #     }
// #    
// #    
// # };
// # int main() {
// #     BlockList bl;
// #     bl.insert('a', 10);
// #     bl.insert('b', 10);
// #     bl.insert('c', 10);
// #     bl.insert('d', 10);
// #     bl.insert('e', 1);
// #         bl.insert('e', 1);
// #         bl.insert('e', 1);
// #         bl.insert('e', 1);
// #     bl.insert('e', 1);
// #     bl.print();
// #     cout << bl.get(7) << endl;
// #     bl.erase(7);
// #     bl.print();
// #     bl.erase(7);
// #     bl.erase(4);
// #     bl.erase(4);
// #     bl.print();
// #     bl.erase(3);
// #     bl.print();
// # }

# https://www.1point3acres.com/bbs/thread-743471-1-1.html

72
电面： 设计键值对存储   单机可恢复注意多线程问题
VO：1 LLVM 看起来很简单没准备 结果挂了。。
       2 设计股票交易中台  输入除了股票 股数外还有一个deadline  主要是设计怎样存储和处理deadline  一个有意思的点是有可能尽管qps没有增加 但是有一段时间的人都选择了在同一时间点deadline的trading，怎样发现和处理
       3 设计单机多线程爬虫 好好看下threadpool怎么用就好
       4  给一个list ipcidr和deny/allow 问一个ip能不能通过。followup是问一个ipcidr能不能通过 不用写 说一下怎么merge就好。
挂在了coding楼主工作已经很少写码了 看了看面经也‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌没觉得难于是眼高手低了。。。LLVM写的太慢没来得及followup  
第四轮的沙雕面试官吐槽我没有跑更多测试用例而是只写了注释应该测哪些所以后来者要注意他家很注重这些。
# https://www.1point3acres.com/bbs/thread-737564-1-1.html

73.
利口 士气
"""
public class Solution {
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        if (digits.length() == 0) {
            return result;
        }
        String[][] letters = { {},{}, {"a", "b", "c"}, {"d", "e", "f"},
        {"g", "h", "i"}, {"j", "k", "l"}, {"m", "n", "o"}, 
        {"p", "q", "r", "s"}, {"t", "u", "v"}, {"w", "x", "y", "z"}};
        helper(result, "", 0, digits, letters);
        return result;
    }
    
    private void helper(List<String> result, String path, int pos, String digits, String[][] letters) {
        if (pos == digits.length()) {
            result.add(path);
            return;
        }
        for (int i = 0; i < letters[digits.charAt(pos) - '0'].length; i++) {
            helper(result, path + letters[digits.charAt(pos) - '0'][i], pos + 1, digits, letters);
        }
    }
}   
"""

74.
利口 其寺耳
气定神闲的小哥 要求O(1) space 刷题不够深入 挂了
"""
// use dfs and record 4 attributes on each dfs result
// closetLeafNode
// closetLeafNodeDist
// hasTarget - the node just visited contains target?
// distToTarget - if contain target, the distance to the target

// also store the closet dist target to leaf and closet leaf node to target globally
// if the node we are checking is target
// closet dist target to leaf = closetLeafNodeDist of visiting node
// closet leaf node to target = closetLeafNode of visiting node
// if one of the node's children contains target
// set distance to target and check
// if the distance to target + closetLeafNodeDist of visiting node < closet dist target to leaf 
// (in here, it is impossible that this is true but target and closet leaf on same child)

// time complexity: O(N)
// space complexity: O(logN) - height of the tree

class Solution {
    class DfsResult {
        TreeNode leafNode;
        int closetLeafDist;
        boolean hasTarget;
        int distToTarget;
        
        public DfsResult(TreeNode leaf, int closetLeafDist, boolean hasTarget, int distToTarget) {
            this.leafNode = leaf;
            this.closetLeafDist = closetLeafDist;
            this.hasTarget = hasTarget;
            this.distToTarget = distToTarget;
        }
    }
    
    private int shortestDist = Integer.MAX_VALUE;
    private TreeNode shortestLeaf = null;
    
    public int findClosestLeaf(TreeNode root, int k) {
        dfsTravel(root, k);
        return shortestLeaf.val;
    }
    
    public DfsResult dfsTravel(TreeNode root, int target) {
        TreeNode closetLeaf = null;
        int closetLeafDist = Integer.MAX_VALUE;
        boolean hasTarget = false;
        int distToTarget = Integer.MAX_VALUE;
        
        if (root == null) {
            return new DfsResult(closetLeaf, closetLeafDist, hasTarget, distToTarget);
        }
        
        DfsResult leftResult = dfsTravel(root.left, target);
        DfsResult rightResult = dfsTravel(root.right, target);
        
        // find closet leaf and leaf dist for root
        if (root.left == null && root.right == null) {
            // root is the leaf
            closetLeaf = root;
            closetLeafDist = 0;
        } else {
            closetLeaf = leftResult.closetLeafDist < rightResult.closetLeafDist ? leftResult.leafNode : rightResult.leafNode;
            closetLeafDist = Math.min(leftResult.closetLeafDist, rightResult.closetLeafDist) + 1;
        }
        
        // check if root is target
        // or if substree has target
        if (root.val == target) {
            hasTarget = true;
            distToTarget = 0;
            
            shortestDist = closetLeafDist;
            shortestLeaf = closetLeaf;
        } else if (leftResult.hasTarget || rightResult.hasTarget) {
            // if yes, calculate the distToTarget, and find closet leaf
            distToTarget = leftResult.hasTarget ? leftResult.distToTarget + 1 : rightResult.distToTarget + 1;
            hasTarget = true;
            
            if (distToTarget + closetLeafDist < shortestDist) {
                shortestDist = distToTarget + closetLeafDist;
                shortestLeaf = closetLeaf;
            }
        }
        
        return new DfsResult(closetLeaf, closetLeafDist, hasTarget, distToTarget);
    }
}

"""




75. LLVM
"""
;; To run: gcc division.ll && ./a.out

; The first two statements declare a string and a function that are used for printing to stdout. You can ignore these.
@.str = private constant [12 x i8] c"Output: %d\0A\00"
@.zerostatement = private constant [15 x i8] c"denom is zero\0A\00"
declare i32 @printf(i8*, ...)

; In this problem, we will be implementing a simple division algorithm in LLVM,
; which is an assembly-like language.

; You will need to understand the following commands:

; Memory: alloca, store, load
; Arithmetic: add, sub
; Conditionals: icmp [integer compare], br [branch]

; Language Reference: http://llvm.org/docs/LangRef.html

; https://tio.run/#llvm


define i32 @convertopositive(i32 %number, i1 %ispositive) {
br i1 %ispositive, label %returnpositive, label %returnnegative

returnnegative:
%positivenumber = sub i32 0, %number
ret i32 %positivenumber

returnpositive:
ret i32 %number
}

define i32 @flipresult(i32 %result, i1 %iscurnumpositive, i1 %iscurdenompositive) {
%sameside = icmp eq i1 %iscurnumpositive, %iscurdenompositive

br i1 %sameside, label %returnpositive, label %returnnegative

returnpositive:
ret i32 %result

returnnegative:
%negativenumber = sub i32 0, %result
ret i32 %negativenumber
}

define i32 @flipremain(i32 %result, i1 %iscurnumpositive) {
br i1 %iscurnumpositive, label %returnpositive, label %returnnegative

returnpositive:
ret i32 %result

returnnegative:
%negativenumber = sub i32 0, %result
ret i32 %negativenumber
}

define i32 @main() {
start:
; Convenience: %str can be used for printing.
%str = getelementptr inbounds [12 x i8], [12 x i8]* @.str, i32 0, i1 0
%zerostatement = getelementptr inbounds [15 x i8], [15 x i8]* @.zerostatement, i32 0, i1 0

; Input: numerator & denominator, as registers.
%num = add i32 0, 23
%denom = add i32 0, 10

; Jump to start of your code.
; Note that there is no fall-through; we must jump to a label or return.
br label %code

; You do not need to modify code above here.
code:
; need to check if denom is 0
%cond_zero = icmp eq i32 0, %denom

br i1 %cond_zero, label %printzero, label %continue

continue:
; note to check negative
%isnumpositive = icmp slt i32 0, %num
%isdenompositive = icmp slt i32 0, %denom

; convert all input to positive
%numpositive = call i32 (i32, i1) @convertopositive(i32 %num, i1 %isnumpositive)
%denompositive = call i32 (i32, i1) @convertopositive(i32 %denom, i1 %isdenompositive)

;call i32 (i8*, ...) @printf(i8* %str, i1 %isnumpositive)
;call i32 (i8*, ...) @printf(i8* %str, i1 %isdenompositive)

; init remain as input
%remain = alloca i32
store i32 %numpositive, i32* %remain

; init count as 0
%count = alloca i32
store i32 0, i32* %count

br label %division

division:
; check if remain is bigger than denom, if so proceed to sub otherwise reach end
%compare = load i32, i32* %remain
%cond = icmp uge i32 %compare, %denompositive

br i1 %cond, label %process, label %print

process:
; sub the remain with denom and increase %count
%total = load i32, i32* %remain
%result = sub i32 %total, %denompositive
store i32 %result, i32* %remain

%count_reg = load i32, i32* %count
%count_update = add i32 1, %count_reg
store i32 %count_update, i32* %count

br label %division

print:
%quotient = load i32, i32* %count
%remainder = load i32, i32* %remain

%actualquotient = call i32 (i32, i1, i1) @flipresult(i32 %quotient, i1 %isnumpositive, i1 %isdenompositive)
%actualremainder = call i32 (i32, i1) @flipremain(i32 %remainder, i1 %isnumpositive)

call i32 (i8*, ...) @printf(i8* %str, i32 %actualquotient)
call i32 (i8*, ...) @printf(i8* %str, i32 %actualremainder)

br label %end

printzero:
call i32 (i8*, ...) @printf(i8* %zerostatement)

br label %end

end:
ret i32 1
}
"""


76. Coding是个新题，具体我也很难解释，
简单来说就是需要为我们的server写一个类似getaway处理所有request，当server连续return多次failure之后， 
gateway不会试图继续request server，而是直接return rejection。多次连续rejection之后再去尝试request server。
然后假设我们有一个primary和一个backup两个server，需要写另一个interface先去试primary然后试backup。
关键是这个题最后要写成类似mock的形式，题目也很复杂看得头晕，问了一大堆clarification，最后勉强跑完了一个test case。
建议各‍‍‌‍‌‍‌‌‌‌‌‌‌‍‍‌‌位好好看一下自己语言lambda function或者function object的syntax。
https://www.1point3acres.com/bbs/thread-881890-1-1.html


"""
class ServerCircuitBreakOpenException(message: String) : Exception(message)

interface CircuitBreaker<Request, Response> {
    fun request(request: Request): Response
}

// how to make this thread safe??
// use synchronized to lock the count we want to change

// when request call
// if map[serverid] >= k reject -> rejectMap[serverid] += 1
// -- if rejectMap[serverid] >= k -> rejectMap[serverid] = 0, map[serverid] = 0
// if pass -> reset all maps for server id
// if fail -> map[serverid] += 1
class SimpleCircuitBreaker(
        private val breakCount: Int,
        primary: ServerClient,
        replicas: List<ServerClient>
) : CircuitBreaker<Int, Int> {
    private val servers: List<ServerClientWrap>

    init {
        servers = listOf(ServerClientWrap(primary, ServerCallCount(0, 0))) +
                replicas.map { client -> ServerClientWrap(client, ServerCallCount(0, 0)) }
    }

    override fun request(request: Int): Int {
        var error: Throwable? = null

        servers.forEach { clientWrap ->
            try {
                return callClient(request, clientWrap.client, clientWrap.callCount)
            } catch (circuitOpenException: ServerCircuitBreakOpenException) {
                // circuit breaker open, continue to next one
                println(circuitOpenException.message)
            } catch (e: Throwable) {
                // request fail, continue to next
                error = e
            }
        }

        throw error ?: ServerCircuitBreakOpenException("All circuit breaker open.")
    }

    private fun callClient(request: Int, client: ServerClient, callCount: ServerCallCount): Int {
        synchronized(callCount) {
            println("${client.id} - $callCount - req$request - thread ${Thread.currentThread().name}")
            if (callCount.fail >= breakCount) {
                // breaker open already
                callCount.reject += 1
                if (callCount.reject >= breakCount) {
                    resetCount(callCount)
                }

                throw ServerCircuitBreakOpenException("client ${client.id} circuit breaker open")
            }
        }

        val response = try {
            val resp = client.call(request)
            // if succeeds, reset all map
            resetCount(callCount)

            resp
        } catch (e: Throwable) {
            synchronized(callCount) {
                callCount.fail += 1
            }

            throw e
        }

        return response
    }

    private fun resetCount(callCount: ServerCallCount) {
        synchronized(callCount) {
            callCount.fail = 0
            callCount.reject = 0
        }
    }
}

class ServerClient(val id: String) {
    fun call(input: Int): Int {
        if (input % 10 == 1) {
            throw Exception("oh no, error on $id")
        } else if (id == "2" && input % 10 == 2) {
            throw Exception("oh no, error-2 on $id")
        } else if (id == "1" && input % 10 == 3) {
            throw Exception("oh no, error-3 on $id")
        }
        return input + 100
    }
}

data class ServerClientWrap(val client: ServerClient, val callCount: ServerCallCount)

data class ServerCallCount(@Volatile var fail: Int, @Volatile var reject: Int)

"""




























"======================================================================"
yimusanfendi summary. 
- 1 刷题网311. Sparse Matrix Multiplication
- 2 Wall And Gates
You are given a m x n 2D grid initialized with these three possible values.
- 1 - A wall or an obstacle. 0 - A gate. INF - Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647. Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

For example, given the 2D grid:
INF  -1  0  INF
INF INF INF  -1
INF  -1 INF  -1
  0  -1 INF INF

After running your function, the 2D grid should be:
  3  -1   0   1
  2   2   1  -1
  1  -1   2  -1
  0  -1   3   4

- 3 刷题网 317 Shortest Distance from All Buildings
- 4 top k elements coming in a stream
# **Explanation**: Let's assume we have a stream of arrays, and the following assumption still 
holds true that k will always be within the range [1,unique number of elements in the array].
# Lets's take the following operations and K=2 
a) Add 1 b) Add 1 c) Add 2 d) Find top 2 elements
e) Add 3 f) Find top 2 elements g) Add 2h) Find top 2 elements
# **For operation a, b and c**, we add the values in heap - 
it's a min heap, so heap would have "1" and "2" element.Also, priority of heap is the frequency of each element.
So presentInHeap map: [1 : 2, 2:1]
1:2 -> means "1" is added and its frequency is 2
2:1 -> means "2" is added and its frequency is 1
**For operation d** - 
we can print top 2 element from the heap
**For operation e**- "3" is added in the map but 2 will be popped out since the heap size which becomes 3
 now exceeds k=2
 So now, we will delete "2" from the main heap but maintain the notInHeap map with popped valuenotInHeap map: [2 :1] , 
 it means that when 2 was popped out from main heap, its frequency so far encountered is 1.
 **For operation f** - Top 2 elements would be "1" and "3"
 **For operation g** - Add "2", since 2 is not there in the heap, hence it add the element in the heap, 
 by getting the frequency from notInHeap map
# ```
# presentInHeap.put(element,notInHeap.getOrDefault(element,0) + 1);
# ```
# This gives the final frequency as 2 for "2" value.So now heap has total three elements:1 with frequency 22 with frequency 23 with frequency 1
# So now, "3" gets popped out from main heap and pushed in notInHeap map
# **For operation h**: find top 2 elements from the heap which is "1" and "2".

-5 刷题网Design 981. Time Based Key-Value Store

-6 MockHashMap
# class mockHashMap:
#     def __init__(self):
#         self.res_dict = {}
#         self.start_time = time.time()
#         self.putCallCount = 0
#         self.putCallTrack = [] # Each Element in the list is the call times in ith 5 minutes
#         self.getCallCount = 0
#         self.getCallTrack = []
#     def put(self, key, val):
#         if key not in res_dict:
#             res_dict[key] = []
#             res_dict[key].append(val)
#         else:
#             res_dict[key].append(val)
#         if (time.time()-start_time)%300 == 0:
#             self.putCallTrack.append(self.putCallCount)
#             self.putCallCount = 0
#         self.putCallCount += 1
#         
#     def get(self, key):
#         if (time.time()-start_time)%300 == 0:
#             self.getCallTrack.append(self.getCallCount)
#             self.putCallCount = 0
#         self.getCallCount += 1   
#         return res_dict[key]
#    
#     def measure_put_load():
#         last_5_min_call = self.putCallCount[-1]
#         return last_5_min_call/300
#         
#     def measure_get_load():
#         last_5_min_call = self.getCallCount[-1]
#         return last_5_min_call/300

-7 Rope for string concatenance

// C++ program to concatenate two strings using
// rope data structure.

-8 BinarySearchTree insert/delete

-9 Find First Missing postive
# https://www.1point3acres.com/bbs/thread-867263-1-1.html


"""
public class Solution {
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        explore(root, result);
        return result;
        // HashMap<Integer, List<Integer>> colToVertical = new HashMap<>();
        // HashMap<TreeNode, Integer> nodeToCol = new HashMap<>();
        // Queue<TreeNode> level = new LinkedList<>();
        // int mostLeft = 0;
        // level.offer(root);
        // nodeToCol.put(root, 0);
        // while (!level.isEmpty()) {
        //     TreeNode node = level.poll();
        //     int curCol = nodeToCol.get(node);
        //     mostLeft = Math.min(mostLeft, curCol);
        //     if (!colToVertical.containsKey(curCol)) {
        //         colToVertical.put(curCol, new ArrayList<Integer>());
        //     }
        //     colToVertical.get(curCol).add(node.val);
        //     if (node.left != null) {
        //         level.offer(node.left);
        //         nodeToCol.put(node.left, curCol - 1);
        //     }
        //     if (node.right != null) {
        //         level.offer(node.right);
        //         nodeToCol.put(node.right, curCol + 1);
        //     }
        // }
        // while (colToVertical.containsKey(mostLeft)) {
        //     result.add(colToVertical.get(mostLeft++));
        // }
        // return result;
        
    }
    private void explore(TreeNode root, List<List<Integer>> result) {
        if (root == null) {
            return;
        }
        List<List<Integer>> pos = new ArrayList<>();
        List<List<Integer>> neg = new ArrayList<>();
        Queue<Pair> level = new LinkedList<>();
        level.offer(new Pair(root, 0));
        while (!level.isEmpty()) {
            Pair pair = level.poll();
            TreeNode node = pair.node;
            if (pair.index >= 0) {
                insert(pos, pair.index, node);
            }
            else {
                insert(neg, -pair.index - 1, node);
            }
            if (node.left != null) {
                level.offer(new Pair(node.left, pair.index - 1));
            }
            if (node.right != null) {
                level.offer(new Pair(node.right, pair.index + 1));
            }
        }
        for (int i = neg.size() - 1; i >= 0; i--) {
            result.add(neg.get(i));
        }
        result.addAll(pos);
    }
    
    private void insert(List<List<Integer>> list, int index, TreeNode node) {
        if (index == list.size()) {
            list.add(new ArrayList<Integer>());
        }
        list.get(index).add(node.val);
    }
    class Pair {
        TreeNode node;
        int index;
        public Pair(TreeNode node, int index) {
            this.node = node;
            this.index = index;
        }
    }
}
"""










