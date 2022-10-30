package com.company;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Databricks {
    interface HtmlParser {
      public List<String> getUrls(String url);
    }
    public List<String> crawl(String startUrl, HtmlParser htmlParser) {
        int idx = startUrl.indexOf('/', 7);
        String hostname = (idx != -1) ? startUrl.substring(0, idx) : startUrl;
        Set<String> visited = ConcurrentHashMap.newKeySet();
        return crawl(startUrl, htmlParser, hostname, visited)
                .collect(Collectors.toList());
    }
    private Stream<String> crawl(String startUrl, HtmlParser htmlParser, String hostname, Set<String> visited) {
        Stream<String> stream = htmlParser.getUrls(startUrl)
                .parallelStream()
                .filter(url -> isSameHostname(url, hostname))
                .filter(url -> visited.add(url))
                .flatMap(url -> crawl(url, htmlParser, hostname, visited));
        return Stream.concat(Stream.of(startUrl), stream);
    }

    private boolean isSameHostname(String url, String hostname){
        if (!url.startsWith(hostname)){
            return false;
        }
        return url.length() == hostname.length() || url.charAt(hostname.length()) == '/';
    }


//    public List<String> crawl(String startUrl, HtmlParser htmlParser){
//        String hostName = getHostName(startUrl);
//        List<String> res = new ArrayList<>();
//        Set<String> visited = new HashSet<>();
//        BlockingQueue<String> queue = new LinkedBlockingDeque<>();
//        Deque<Future> tasks = new ArrayDeque<>();
//        queue.offer(startUrl);
//        ExecutorService excuter = Executors.newFixedThreadPool(4, r -> {
//            Thread t = new Thread(r);
//            t.setDaemon(true);
//            return t;
//        });
//        while (true){
//            String url = queue.poll();
//            if (getHostName(url).equals(hostName) && !visited.contains(url)){
//                tasks.add(excuter.submit( () -> {
//                    List<String> newUrls = htmlParser.getUrls(url);
//                    for
//                }))
//            }
//        }
//    }
//    private String getHostName(String url){
//      url = url.substring(7);
//      String[] parts = url.split("/");
//      return parts[0]
//    }
    static class MyIterator implements Iterator<Integer> {
        Map<Integer,
                Integer> snapshot;
        Map<Integer,
                Integer> curset;
        Iterator<Integer> it;

        MyIterator(Map<Integer, Integer> s, Map<Integer, Integer> c) {
            this.snapshot = s;
            this.curset = c;
            it = s.keySet().iterator();
        }

        @Override
        public boolean hasNext() {
            boolean res = it.hasNext();
            if (res == false) snapshot.clear();
            return res;
        }

        @Override
        public Integer next() {
            Integer res = it.next();
            if (snapshot.get(res) == 0) curset.put(res, 0);
            return res;
        }
    }

    public static class SnapshotSet {
        Map<Integer, Integer> snapshot = new HashMap<Integer, Integer>();
        Map<Integer, Integer> curset = new HashMap<Integer, Integer>();

        void add(Integer e) {
            if (snapshot.containsKey(e)) snapshot.put(e, 0); // not deleted
            else curset.put(e, 0);
        }

        void remove(Integer e) {
            if (snapshot.containsKey(e)) snapshot.put(e, 1); // deleted
            else curset.remove(e);
        }

        boolean contains(Integer e) {
            return snapshot.containsKey(e) && snapshot.get(e) == 0 || curset.containsKey(e);
        }

        Iterator<Integer> iterator() {
            snapshot = curset;
            curset = new HashMap<Integer, Integer>();
            return new MyIterator(snapshot, curset);
        }

        public static void main(String[] args) {
            SnapshotSet ss = new SnapshotSet();
            ss.add(8);
            ss.add(2);
            ss.add(5);
            ss.remove(5);
            Iterator<Integer> it = ss.iterator();
            System.out.println(ss.contains(2));
            ss.add(1);
            ss.remove(2);
            System.out.println(ss.contains(2));
            while (it.hasNext()) {
                System.out.print(it.next() + ",");
            }
            System.out.println();
            ss.add(5);
            it = ss.iterator();
            while (it.hasNext()) {
                System.out.print(it.next() + ",");
            }
            System.out.println(ss.contains(2));
        }
    }
}
